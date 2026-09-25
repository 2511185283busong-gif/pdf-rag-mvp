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

## Results

Evaluation configuration:

- 36 chunks from a 101-page lecture.
- 10 manually verified questions.
- `intfloat/multilingual-e5-small` for dense recall.
- `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` for reranking.
- Top 3 final results.

| Candidate K | Method | Hit@3 | MRR@3 |
| ---: | --- | ---: | ---: |
| 10 | Dense embedding only | 0.70 | 0.60 |
| 10 | Dense embedding plus rerank | 0.90 | 0.80 |
| 20 | Dense embedding plus rerank | 1.00 | 0.7833 |
| 36 | Dense embedding plus rerank | 1.00 | 0.80 |

Increasing Candidate K gives reranking more recall candidates to inspect. On
this 36-chunk lecture, K=20 is sufficient to place evidence for every labelled
question in the final Top 3. K=36 is a diagnostic full-rerank setting, not a
practical default for larger collections.

K=20 improves coverage over K=10, but its MRR is slightly lower because some
correct chunks move from rank 1 to rank 2 or 3. Candidate K is therefore a
recall-versus-ranking-cost tradeoff, not a value that always improves every
metric.

The Filtering limitation case accepts both chunks 0002 and 0003: the relevant
slide is page 14, which occurs at their chunk boundary. This keeps the labels
aligned with the evidence rather than treating a valid overlap hit as a miss.

This is a small benchmark result rather than a general performance claim.

## Worked Examples

The following examples are recorded from local runs. They illustrate specific
behaviors; they are not aggregate performance claims.

### 1. Lexical baseline versus dense embedding

Using the three chunks from `resume2_chunks.json`, the query below has no
token overlap with the English resume text:

```text
How are useful passages selected for a question?
```

TF-IDF therefore produced an all-zero tie across the original chunks. Dense
embedding with `intfloat/multilingual-e5-small` placed
`resume2_chunk_0001` first (cosine `0.776867`), which contains the candidate's
RAG retrieval and semantic-search learning material. This is an example of
dense retrieval helping with a wording mismatch, not proof that it is always
better.

The synthetic keyword distractor included by the experiment also ranked ahead
of the real chunks for this query. That result is deliberately retained: dense
retrieval can still be misled, which is why the production path uses reranking
and the evaluation below rather than relying on one example.

Reproduce the comparison:

```bash
.venv/bin/python scripts/tfidf_wording_experiment.py \
  chunks/resume2_chunks.json \
  --query "How are useful passages selected for a question?"

.venv/bin/python scripts/embedding_wording_experiment.py \
  chunks/resume2_chunks.json \
  --query "How are useful passages selected for a question?"
```

### 2. Reranking a recalled candidate

For the manually labelled lecture question below, the correct evidence is
`Lecture 8b Distributed Constraints_chunk_0014` (pages 36-39):

```text
What are the roles of ok, nogood, and add-neighbor messages in asynchronous backtracking (ABT)?
```

With `candidate-k=10`, embedding-only Top 3 returned chunks `0021`, `0015`,
and `0022`, so the correct chunk was not visible in Top 3. It was still inside
the 10 recalled candidates. The cross-encoder reranker moved chunk `0014` to
rank 1. This illustrates the intended division of work: dense retrieval
provides candidate coverage, then reranking performs a more precise
query-chunk comparison.

Run the full 10-question report:

```bash
.venv/bin/python scripts/evaluate_retrieval.py \
  "chunks/Lecture 8b Distributed Constraints_chunks.json" \
  evals/distributed_constraints_cases.json \
  --candidate-k 10 \
  --top-k 3
```
