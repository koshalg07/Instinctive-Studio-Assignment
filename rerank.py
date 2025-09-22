# rerank.py
import os
import sqlite3
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
import joblib

# Paths
INDEX_PATH = "data/faiss_index.bin"
MAPPING_PATH = "data/id_mapping.npy"
DB_PATH = "data/chunks.db"

# Load learned reranker if exists, else fall back to cross-encoder
LEARNED_PATH = "data/reranker_lr.joblib"
lr_model = None
if os.path.exists(LEARNED_PATH):
    lr_model = joblib.load(LEARNED_PATH)
else:
    reranker = CrossEncoder("cross-encoder/ms-marco-electra-base")

# Load retriever (same model used for FAISS index build!)
retriever = SentenceTransformer("all-mpnet-base-v2")

# Fetch top-K candidates from FAISS
def fetch_candidates_faiss(query, top_k=20):
    # Load FAISS index + ID mapping
    index = faiss.read_index(INDEX_PATH)
    id_mapping = np.load(MAPPING_PATH)

    # Encode query with retriever
    q_emb = retriever.encode([query], normalize_embeddings=True)
    q_emb = np.array(q_emb).astype("float32")

    # Search FAISS
    D, I = index.search(q_emb, top_k)
    ids = id_mapping[I[0]]

    # Fetch chunk text from SQLite
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    results = []
    for score, idx in zip(D[0], ids):
        cur.execute("SELECT chunk_text FROM chunks WHERE chunk_id=?", (int(idx),))
        row = cur.fetchone()
        if row:
            results.append((idx, score, row[0]))
    conn.close()
    return results

# Rerank with cross-encoder
def rerank(query, candidates):
    # candidates: list of (chunk_id, base_score, text)
    if lr_model is not None:
        # Use learned logistic regression with features: [vector_score, bm25_score]
        # Compute BM25 via FTS table for given candidate chunk_ids
        from features import bm25_scores  # lazy import
        chunk_ids = [int(cid) for cid, _, _ in candidates]
        id_to_bm25 = bm25_scores(query, chunk_ids)

        reranked = []
        for chunk_id, base_score, text in candidates:
            bm25 = id_to_bm25.get(int(chunk_id), 0.0)
            X = np.array([[float(base_score), float(bm25)]], dtype=np.float32)
            prob = float(lr_model.predict_proba(X)[0, 1])
            reranked.append({
                "chunk_id": int(chunk_id),
                "base_score": float(base_score),
                "bm25_score": float(bm25),
                "rerank_score": prob,
                "text": text[:300] + ("..." if len(text) > 300 else ""),
            })
        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        return reranked
    else:
        # Cross-encoder fallback
        pairs = [(query, text) for (_, _, text) in candidates]
        scores = reranker.predict(pairs)
        reranked = []
        for (chunk_id, base_score, text), new_score in zip(candidates, scores):
            reranked.append(
                {
                    "chunk_id": int(chunk_id),
                    "base_score": float(base_score),
                    "rerank_score": float(new_score),
                    "text": text[:300] + ("..." if len(text) > 300 else ""),
                }
            )
        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        return reranked

# Demo
if __name__ == "__main__":
    query = "What protective equipment is recommended for machine operators?"
    print("\nFetching candidates with FAISS...")
    candidates = fetch_candidates_faiss(query, top_k=20)
    print(f"Got {len(candidates)} candidates")

    print("\nReranking...")
    reranked = rerank(query, candidates)

    for r in reranked[:5]:
        print(
            f"[{r['rerank_score']:.4f}] chunk_id={r['chunk_id']} "
            f"(base {r['base_score']:.4f}) -> {r['text']}"
        )
