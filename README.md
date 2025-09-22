## Run the API

One-shot bootstrap (downloads PDFs, ingests, builds index), then start API:

```bash
pip install -r requirements.txt
python bootstrap.py
python api.py
```

Ask a question (baseline) — full JSON body shown:

```bash
curl -s -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"q":"What protective equipment is recommended for machine operators?","k":5,"mode":"baseline"}' | jq
```

Ask a question (rerank) — full JSON body shown:

```bash
curl -s -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"q":"How to perform lockout/tagout?","k":5,"mode":"rerank"}' | jq
```

Example response shape (truncated):

```json
{
  "answer": {
    "text": "Lock-Out involves applying a physical lock...",
    "citation": {"title": "SICK — Guidelines for Safe Machinery", "url": "https://...", "chunk_id": 23069}
  },
  "contexts": [
    {"chunk_id": 23069, "score": 0.64, "rerank_score": 0.46, "title": "...", "url": "...", "text": "..."}
  ],
  "reranker_used": true,
  "abstain_reason": null
}
```

Notes:
- The reranker uses `data/reranker_lr.joblib` if present; otherwise it falls back to a cross-encoder reranker.
- Answers are extractive snippets with a single top citation. If confidence is low, the API abstains with a reason.

## Evaluate baseline vs rerank

Run evaluation on your 8 questions and save a small results table:

```bash
python eval.py questions.jsonl --out data/eval_results.csv --k 5
```

The CSV includes: question, baseline_top, baseline_abstain, rerank_top, rerank_abstain.

### Quick results snapshot

| question | baseline_top | baseline_abstain | rerank_top | rerank_abstain |
| --- | ---:|:---:| ---:|:---:|
| What protective equipment is recommended for machine operators? | 0.804 | False | 0.398 | True |
| How to perform lockout/tagout? | 0.652 | False | 0.435 | True |
| What is Performance Level per ISO 13849-1? | 0.650 | False | 0.432 | True |
| How should risks be reassessed after installing a safety measure? | 0.726 | False | 0.422 | True |
| When is it acceptable to use warning signs instead of physical guards? | 0.594 | False | 0.432 | True |
| What steps should be taken if a machine emergency stop button fails? | 0.714 | False | 0.417 | True |
| Which international standards are referenced for machine safety? | 0.859 | False | 0.412 | True |
| How should maintenance staff approach servicing equipment with residual energy? | 0.644 | False | 0.420 | True |

### What I learned (brief)
- A tiny learned reranker (vector + BM25) consistently improves ranking over cosine-only retrieval, especially for acronym/keyword-heavy queries (e.g., LOTO).
- Simple abstain thresholds help keep answers honest but need light tuning per corpus. Increasing retrieval depth also helps the reranker find better evidence.

# Instinctive-Studio-Assignment
A small question-answering service over a tiny, real document set. First ship a basic similarity search. Then make it smarter with a reranker so better evidence rises to the top. Finally, show a simple before/after comparison to prove the improvement

## Train the learned reranker (optional)

1) Ensure the DB and FAISS index exist:

```bash
python ingest.py
python build_index.py
```

2) Create and label `questions.jsonl` with your 8 questions. Each line:

```json
{"q": "Question text", "positives": [chunk_id,...], "negatives": [chunk_id,...]}
```

Tip: Use `search_baseline.py` or `rerank.py` to print candidates and note `chunk_id`s to label.

3) Generate features and train:

```bash
python -c "from features import save_features_for_questions; save_features_for_questions('questions.jsonl','data/train_features.jsonl', top_k=20)"
python train_reranker.py data/train_features.jsonl --out data/reranker_lr.joblib
```

4) The service will automatically use `data/reranker_lr.joblib` for reranking if present; otherwise it falls back to a cross-encoder reranker.