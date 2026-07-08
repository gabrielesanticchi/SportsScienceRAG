# Evaluation gold set

`gold.jsonl` — one JSON object per line:
`{"query": str, "relevant": {bare_filename: grade}, "notes": str}`

## Grading rubric (document-level relevance)

- **3 — primary source:** the paper is centrally about the query topic; the
  paper you would cite first.
- **2 — substantive:** the topic is a real, developed section of the paper.
- **1 — peripheral:** the topic is mentioned or touched only in passing.
- **unlabeled (0):** not relevant — omit the paper from the query's `relevant` map.

Scoring is document-level: a paper is identified by its bare filename
(one paper per `source` in the collection). Metrics (via `ranx`):
`mrr`, `recall@5`, `recall@10`, `hit_rate@1`, `ndcg@10`.

## Corpus

31 graded queries over 26 papers in the `sport-science-documents` collection
(2096 points). Every corpus paper is referenced by at least one query. Mostly
known-item (single grade-3 target) with several topical multi-paper queries.
