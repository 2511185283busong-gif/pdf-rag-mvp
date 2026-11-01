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

- 32 chunks from a 101-page lecture.
- 10 manually verified questions, all in Chinese against an English lecture.
- `intfloat/multilingual-e5-small` for dense recall.
- `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` for reranking.

| Candidate K | Method | Final sources | Hit | MRR |
| ---: | --- | ---: | ---: | ---: |
| 10 | Dense embedding only | 3 | 0.60 | 0.60 |
| 10 | Dense embedding plus rerank | 3 | 1.00 | 0.95 |
| 20 | Dense embedding plus rerank | 3 | 1.00 | 0.90 |
| 32 | Dense embedding plus rerank | 3 | 1.00 | 0.90 |
| 20 | Dense embedding plus rerank (answer default) | 2 | 1.00 | 0.90 |
| 20 | Dense embedding plus rerank | 1 | 0.80 | 0.80 |

K=32 is a diagnostic full-rerank setting, not a practical default for larger
collections. On this lecture K=10 already places all evidence in the Top 3, and
its MRR is higher than at K=20. Candidate K is therefore a recall-versus-ranking
tradeoff, not a value that always improves every metric.

The ABT completeness case accepts both chunks 0016 and 0017: the theorem on page
50 occurs at their chunk boundary. This keeps the labels aligned with the evidence
rather than treating a valid overlap hit as a miss.

### Effect of the extraction and chunking fix

The earlier extraction lost spaces between words, and lost small-capital terms
such as `ok` and `nogood`. Repeated slide headers also entered every chunk. With
the same 10 questions and K=20, retrieval before and after the fix compares as
follows:

| Final sources | Reranked Hit before | Reranked Hit after | Reranked MRR before | Reranked MRR after |
| ---: | ---: | ---: | ---: | ---: |
| 3 | 1.00 | 1.00 | 0.78 | 0.90 |
| 2 | 0.90 | 1.00 | 0.75 | 0.90 |
| 1 | 0.60 | 0.80 | 0.60 | 0.80 |

Dense-only Hit@3 fell from 0.70 to 0.60, so the gain comes from reranking. Chunk
ids changed with the fix, so the labels were re-verified against the new chunks
using the recorded `verified_pages`. The questions are the same, but the two runs
are not a controlled comparison of identical labels.

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
`resume2_chunk_0001` first (cosine `0.776868`), which contains the candidate's
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
`Lecture 8b Distributed Constraints_chunk_0011` (pages 29-37):

```text
异步回溯（ABT）中的 ok、nogood 和 add-neighbor 消息分别有什么作用？
```

With `candidate-k=10`, embedding-only Top 3 returned chunks `0012`, `0018`, and
`0016`, so the correct chunk was not visible in Top 3. It was still inside the 10
recalled candidates. The cross-encoder reranker moved chunk `0011` to rank 1. This
illustrates the intended division of work: dense retrieval provides candidate
coverage, then reranking performs a more precise query-chunk comparison.

Run the full 10-question report:

```bash
.venv/bin/python scripts/evaluate_retrieval.py \
  "chunks/Lecture 8b Distributed Constraints_chunks.json" \
  evals/distributed_constraints_cases.json \
  --candidate-k 10 \
  --top-k 3
```
