# Retrieval Evaluation

## Distributed Constraints Lecture

`distributed_constraints_cases.json` contains 10 manually verified retrieval
cases for `Lecture 8b Distributed Constraints.pdf`.
`distributed_constraints_reworded_cases.json` adds 16 narrower Chinese and English
questions about the same evidence, written to expose wording sensitivity. The
source PDF is local and is not committed to this repository. Each case records the
expected chunks and the original slide page that was checked during labeling.

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

- 18 chunks from a 101-page lecture.
- `intfloat/multilingual-e5-small` for dense recall.
- `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` for reranking.

Original set (10 Chinese questions):

| Candidate K | Method | Final sources | Hit | MRR |
| ---: | --- | ---: | ---: | ---: |
| 10 | Dense embedding only | 3 | 0.80 | 0.47 |
| 10 | Dense embedding plus rerank | 3 | 1.00 | 0.85 |
| 20 | Dense embedding plus rerank | 3 | 1.00 | 0.85 |
| 18 | Dense embedding plus rerank (all chunks) | 3 | 1.00 | 0.85 |
| 20 | Dense embedding plus rerank (answer default) | 2 | 1.00 | 0.85 |
| 20 | Dense embedding plus rerank | 1 | 0.70 | 0.70 |

Reworded set (16 Chinese and English questions):

| Candidate K | Method | Final sources | Hit | MRR |
| ---: | --- | ---: | ---: | ---: |
| 20 | Dense embedding only | 3 | 0.69 | 0.63 |
| 10 | Dense embedding plus rerank | 3 | 0.81 | 0.78 |
| 20 | Dense embedding plus rerank | 3 | 0.94 | 0.88 |
| 20 | Dense embedding plus rerank (answer default) | 2 | 0.94 | 0.88 |
| 20 | Dense embedding plus rerank | 1 | 0.81 | 0.81 |

Candidate K is a recall-versus-ranking tradeoff: K=10 matches K=20 on the original
set but loses evidence on the reworded set. The one remaining reworded miss is
`r-ok-zh`, where the correct chunk is not in the reranked Top 3.

Relevant chunks are the ones whose text contains the recorded evidence sentence,
so a sentence repeated on another slide or kept by chunk overlap can have several
valid chunks. This keeps the labels aligned with the evidence rather than treating
a valid hit as a miss.

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

### Effect of removing formula noise

Lecture slides also produced 582 `(cid:N)` placeholders for glyphs with no Unicode
mapping, and on formula slides they made up about half of a chunk. Removing them
and the 911 lines left without a word or multi-digit number changes retrieval as
follows, with K=20 and relevance taken from the evidence sentences on both sides:

| Question set | Final sources | Hit before | Hit after | MRR before | MRR after |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original | 3 | 1.00 | 1.00 | 0.90 | 0.85 |
| Original | 2 | 1.00 | 1.00 | 0.90 | 0.85 |
| Original | 1 | 0.80 | 0.70 | 0.80 | 0.70 |
| Reworded | 3 | 0.81 | 0.94 | 0.74 | 0.88 |
| Reworded | 2 | 0.75 | 0.94 | 0.72 | 0.88 |
| Reworded | 1 | 0.69 | 0.81 | 0.69 | 0.81 |

The reworded questions improve and the original questions lose some rank, so this
is a tradeoff rather than a uniform gain. Chunks also span more pages: the median
rose from 3 to 5 and the maximum from 10 to 14. The cleanup was chosen using these
same questions, and removing placeholders alone lowered original Hit@2 to 0.90, so
the reworded results are not an independent test.

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
`Lecture 8b Distributed Constraints_chunk_0011` (pages 61-70):

```text
Asynchronous Weak-Commitment Search 是否完备？
```

With `candidate-k=10`, embedding-only Top 3 returned chunks `0009`, `0010`, and
`0008`, so the correct chunk was not visible in Top 3. It was still inside the 10
recalled candidates. The cross-encoder reranker moved chunk `0011` to rank 2. This
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
