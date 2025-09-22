## Run the API

Start the Flask server:

```bash
python api.py
```

Ask a question (baseline):

```bash
curl -s -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"q":"What protective equipment is recommended for machine operators?","k":5,"mode":"baseline"}' | jq
```

Ask a question (rerank):

```bash
curl -s -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"q":"How to perform lockout/tagout?","k":5,"mode":"rerank"}' | jq
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