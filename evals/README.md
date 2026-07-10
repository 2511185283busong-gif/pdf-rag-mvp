# Retrieval Evaluation

## Distributed Constraints Lecture

`distributed_constraints_cases.json` contains 10 manually verified retrieval
cases for `Lecture 8b Distributed Constraints.pdf`. The source PDF is local
and is not committed to this repository. Each case records the expected chunk
and the original slide page that was checked during labeling.

Run the evaluation after creating the lecture chunks:

```bash
.venv/bin/python scripts/evaluate_retrieval.py \
  "chunks/Lecture 8b Distributed Constraints_chunks.json" \
  evals/distributed_constraints_cases.json \
  --candidate-k 10 \
  --top-k 3
```

## Result

Evaluation configuration:

- 36 chunks from a 101-page lecture.
- 10 manually verified questions.
- `intfloat/multilingual-e5-small` for dense recall.
- `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` for reranking.
- Top 10 recall candidates and Top 3 final results.

| Method | Hit@3 | MRR@3 |
| --- | ---: | ---: |
| Dense embedding only | 0.60 | 0.55 |
| Dense embedding plus rerank | 0.80 | 0.70 |

Reranking recovered the correct evidence for the ABT remote-calls and AWC
completeness questions. It did not recover every labelled case, so this is a
small benchmark result rather than a general performance claim.
