# Individual Reflection

## Contribution

I helped complete the evaluation pipeline end to end: golden dataset generation, offline RAG agent behavior, retrieval metrics, multi-judge scoring, regression gate logic, benchmark report generation, and failure analysis.

## Technical Notes

- Hit Rate checks whether at least one expected document appears in the retrieved top-k results.
- MRR rewards retrieving the expected document earlier in the ranking.
- Multi-judge consensus reduces dependence on one scoring rubric by comparing an accuracy judge with a policy/safety judge.
- The release gate should not use score alone; it should also check retrieval quality, agreement, latency, and regression delta.

## Lessons Learned

The most important lesson is that answer quality cannot be debugged without retrieval quality. A correct-looking answer can still be fragile if the retriever found the wrong evidence. Small normalization details such as plural words and related terms can have a large effect on Hit Rate and MRR.

## Next Improvements

- Replace keyword retrieval with BM25 or embeddings.
- Add live LLM judges when API keys are available.
- Add more red-team and multi-turn cases.
- Compare cost/quality trade-offs between full judging and sampled judging.
