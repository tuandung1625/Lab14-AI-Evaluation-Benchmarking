# Failure Analysis Report

## 1. Benchmark Overview

- Agent version: Agent_V2_Optimized
- Total cases: 60
- Pass/Fail: 60/0
- Average LLM-judge score: 4.9267 / 5.0
- Retrieval Hit Rate: 1.0000
- MRR: 1.0000
- Faithfulness: 0.9997
- Relevancy: 0.9390
- Multi-judge agreement rate: 0.9833
- Average latency: 0.0268 seconds/case
- Estimated evaluation cost: 0.0015 USD
- Release gate decision: APPROVE

## 2. Failure Clustering

No V2 cases failed the release threshold. The remaining quality risks are therefore analyzed as near-miss clusters instead of hard failures.

| Cluster | Count | Evidence | Likely Cause |
| --- | ---: | --- | --- |
| Lexical mismatch risk | 0 failed after fix | Earlier dry run missed terms such as `refunds` vs `refund` and `recovery` vs `recoverable`. | Retrieval used exact token overlap before normalization. |
| Role-boundary ambiguity | 0 failed after fix | Questions containing words such as `allowed`, `manage`, and `configure` can attract role permission documents. | Generic permission words can outrank the more specific domain topic. |
| Prompt injection risk | 0 failed after fix | Adversarial cases ask the agent to ignore policy or invent rules. | The generator needs hard cases to ensure the agent refuses unsupported instructions. |
| Cost/latency risk | 0 failed | V2 latency is below the 0.25 second gate and cost is stable. | Offline deterministic judge keeps the benchmark cheap; live LLM judges would need batching/caching. |

## 3. 5 Whys Analysis

### Case A: Prompt injection about refunds

1. Symptom: The user asks the agent to ignore refund policy and write unrelated content.
2. Why 1: A normal generator might follow the latest instruction instead of the retrieved policy.
3. Why 2: The answer composer needs an explicit guard for injection terms such as `ignore`, `invent`, and unrelated creative requests.
4. Why 3: The golden set must include adversarial examples, otherwise this weakness is invisible.
5. Why 4: Retrieval-only metrics cannot detect unsafe instruction following.
6. Root cause: Generation quality and instruction safety require judge metrics in addition to retrieval metrics.

Action taken: The agent now refuses to ignore documentation and the judge includes a policy/safety rubric.

### Case B: Deleted project recovery question

1. Symptom: Role-boundary words can pull `Role Permissions` above `Data Retention`.
2. Why 1: The question contains generic terms such as `allowed`, `manage`, and `configure`.
3. Why 2: Exact keyword matching treats generic policy words and domain-specific words too similarly.
4. Why 3: The retriever lacked normalization for related terms such as `recovery` and `recoverable`.
5. Why 4: The original scaffold did not compare expected document IDs with retrieved IDs.
6. Root cause: Retrieval ranking needs query normalization and ground-truth document ID evaluation.

Action taken: Token normalization and synonym mapping were added; Hit Rate and MRR now verify the retrieval stage.

### Case C: Regression release confidence

1. Symptom: A single average score can hide retrieval regression, latency regression, or judge disagreement.
2. Why 1: The original gate only compared V1 and V2 score delta.
3. Why 2: Quality, cost, and performance are separate product risks.
4. Why 3: A benchmark needs explicit thresholds instead of a manual decision.
5. Why 4: Without a saved baseline, teams cannot explain whether a new version is safer to release.
6. Root cause: Regression testing must combine multiple metrics and persist baseline results.

Action taken: The release gate now checks average score, Hit Rate, agreement rate, latency, and regression delta.

## 4. Improvement Plan

- Add semantic embeddings or BM25 when a real document collection is available.
- Cache live LLM judge calls and sample only risky cases to reduce evaluation cost by at least 30 percent.
- Add Cohen's Kappa or weighted agreement for stricter judge reliability measurement.
- Expand the hard-case set with multi-turn cases, conflicting documents, and out-of-context questions.
- Track per-cluster metrics over time so regressions are easier to explain.
