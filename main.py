import asyncio
import json
import os
import time
from typing import Dict, List, Tuple

from agent.main_agent import MainAgent
from engine.llm_judge import LLMJudge
from engine.retrieval_eval import RetrievalEvaluator
from engine.runner import BenchmarkRunner


QUALITY_GATE = {
    "min_avg_score": 3.5,
    "min_hit_rate": 0.85,
    "min_agreement_rate": 0.65,
    "max_latency_seconds": 0.25,
    "max_regression_drop": -0.05,
}


def load_dataset(path: str = "data/golden_set.jsonl") -> List[Dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}. Run 'python data/synthetic_gen.py' first.")
    with open(path, "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]
    if len(dataset) < 50:
        raise ValueError(f"Golden set must contain at least 50 cases; found {len(dataset)}.")
    return dataset


def summarize(agent_version: str, results: List[Dict], elapsed: float) -> Dict:
    total = len(results)
    total_tokens = sum(item["usage"]["tokens_used"] for item in results)
    total_cost = sum(item["usage"]["estimated_cost_usd"] for item in results)
    pass_count = sum(1 for item in results if item["status"] == "pass")

    metrics = {
        "avg_score": sum(item["judge"]["final_score"] for item in results) / total,
        "hit_rate": sum(item["ragas"]["retrieval"]["hit_rate"] for item in results) / total,
        "mrr": sum(item["ragas"]["retrieval"]["mrr"] for item in results) / total,
        "faithfulness": sum(item["ragas"]["faithfulness"] for item in results) / total,
        "relevancy": sum(item["ragas"]["relevancy"] for item in results) / total,
        "agreement_rate": sum(item["judge"]["agreement_rate"] for item in results) / total,
        "avg_latency_seconds": sum(item["latency"] for item in results) / total,
        "pass_rate": pass_count / total,
        "total_tokens": total_tokens,
        "estimated_cost_usd": round(total_cost, 6),
        "throughput_cases_per_second": round(total / max(elapsed, 0.001), 2),
    }

    return {
        "metadata": {
            "version": agent_version,
            "total": total,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_seconds": round(elapsed, 3),
            "quality_gate": QUALITY_GATE,
        },
        "metrics": {key: round(value, 4) if isinstance(value, float) else value for key, value in metrics.items()},
    }


async def run_benchmark_with_results(agent_version: str, dataset: List[Dict]) -> Tuple[List[Dict], Dict]:
    print(f"Starting benchmark for {agent_version}...")
    started = time.perf_counter()

    runner = BenchmarkRunner(
        agent=MainAgent(version="v1" if "V1" in agent_version else "v2"),
        evaluator=RetrievalEvaluator(),
        judge=LLMJudge(),
    )
    results = await runner.run_all(dataset, batch_size=10)
    elapsed = time.perf_counter() - started
    return results, summarize(agent_version, results, elapsed)


def decide_release(v1_summary: Dict, v2_summary: Dict) -> Dict:
    v1_metrics = v1_summary["metrics"]
    v2_metrics = v2_summary["metrics"]
    delta_score = v2_metrics["avg_score"] - v1_metrics["avg_score"]
    delta_hit_rate = v2_metrics["hit_rate"] - v1_metrics["hit_rate"]
    checks = {
        "avg_score": v2_metrics["avg_score"] >= QUALITY_GATE["min_avg_score"],
        "hit_rate": v2_metrics["hit_rate"] >= QUALITY_GATE["min_hit_rate"],
        "agreement_rate": v2_metrics["agreement_rate"] >= QUALITY_GATE["min_agreement_rate"],
        "latency": v2_metrics["avg_latency_seconds"] <= QUALITY_GATE["max_latency_seconds"],
        "regression_delta": delta_score >= QUALITY_GATE["max_regression_drop"],
    }

    return {
        "decision": "APPROVE" if all(checks.values()) else "BLOCK_RELEASE",
        "checks": checks,
        "delta": {
            "avg_score": round(delta_score, 4),
            "hit_rate": round(delta_hit_rate, 4),
            "estimated_cost_usd": round(v2_metrics["estimated_cost_usd"] - v1_metrics["estimated_cost_usd"], 6),
            "avg_latency_seconds": round(
                v2_metrics["avg_latency_seconds"] - v1_metrics["avg_latency_seconds"], 4
            ),
        },
    }


async def main() -> None:
    try:
        dataset = load_dataset()
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc))
        return

    v1_results, v1_summary = await run_benchmark_with_results("Agent_V1_Base", dataset)
    v2_results, v2_summary = await run_benchmark_with_results("Agent_V2_Optimized", dataset)
    gate = decide_release(v1_summary, v2_summary)

    v2_summary["regression"] = {
        "baseline": v1_summary["metrics"],
        "gate": gate,
    }

    report = {
        "v1": {"summary": v1_summary, "results": v1_results},
        "v2": {"summary": v2_summary, "results": v2_results},
    }

    os.makedirs("reports", exist_ok=True)
    with open("reports/summary.json", "w", encoding="utf-8") as f:
        json.dump(v2_summary, f, ensure_ascii=False, indent=2)
    with open("reports/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\nRegression summary")
    print(f"V1 score: {v1_summary['metrics']['avg_score']}")
    print(f"V2 score: {v2_summary['metrics']['avg_score']}")
    print(f"Hit rate: {v2_summary['metrics']['hit_rate']}")
    print(f"MRR: {v2_summary['metrics']['mrr']}")
    print(f"Agreement: {v2_summary['metrics']['agreement_rate']}")
    print(f"Decision: {gate['decision']}")


if __name__ == "__main__":
    asyncio.run(main())
