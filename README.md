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