from typing import Dict, List


def _token_set(text: str) -> set:
    return {part.strip(".,;:!?()[]").lower() for part in text.split() if part.strip()}


class RetrievalEvaluator:
    def calculate_hit_rate(self, expected_ids: List[str], retrieved_ids: List[str], top_k: int = 3) -> float:
        if not expected_ids:
            return 1.0 if not retrieved_ids else 0.0
        top_retrieved = retrieved_ids[:top_k]
        return 1.0 if any(doc_id in top_retrieved for doc_id in expected_ids) else 0.0

    def calculate_mrr(self, expected_ids: List[str], retrieved_ids: List[str]) -> float:
        if not expected_ids:
            return 1.0 if not retrieved_ids else 0.0
        for index, doc_id in enumerate(retrieved_ids):
            if doc_id in expected_ids:
                return 1.0 / (index + 1)
        return 0.0

    def calculate_answer_relevancy(self, expected_answer: str, answer: str) -> float:
        expected_tokens = _token_set(expected_answer)
        answer_tokens = _token_set(answer)
        if not expected_tokens:
            return 0.0
        return len(expected_tokens.intersection(answer_tokens)) / len(expected_tokens)

    def calculate_faithfulness(self, contexts: List[str], answer: str) -> float:
        context_tokens = _token_set(" ".join(contexts))
        answer_tokens = _token_set(answer)
        if not answer_tokens:
            return 0.0
        return min(1.0, len(answer_tokens.intersection(context_tokens)) / max(1, len(answer_tokens) * 0.65))

    async def score(self, test_case: Dict, response: Dict) -> Dict:
        expected_ids = test_case.get("expected_retrieval_ids", [])
        retrieved_ids = response.get("retrieved_ids", [])
        contexts = response.get("contexts", [])
        answer = response.get("answer", "")

        hit_rate = self.calculate_hit_rate(expected_ids, retrieved_ids)
        mrr = self.calculate_mrr(expected_ids, retrieved_ids)
        relevancy = self.calculate_answer_relevancy(test_case.get("expected_answer", ""), answer)
        faithfulness = self.calculate_faithfulness(contexts, answer)

        return {
            "faithfulness": round(faithfulness, 4),
            "relevancy": round(relevancy, 4),
            "retrieval": {
                "hit_rate": hit_rate,
                "mrr": round(mrr, 4),
                "expected_ids": expected_ids,
                "retrieved_ids": retrieved_ids,
            },
        }

    async def evaluate_batch(self, dataset: List[Dict], responses: List[Dict]) -> Dict:
        scores = [await self.score(case, response) for case, response in zip(dataset, responses)]
        total = len(scores) or 1
        return {
            "avg_hit_rate": sum(item["retrieval"]["hit_rate"] for item in scores) / total,
            "avg_mrr": sum(item["retrieval"]["mrr"] for item in scores) / total,
        }
