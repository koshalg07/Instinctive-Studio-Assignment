import sqlite3
import json
import re
from typing import List, Tuple, Dict

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

DB_PATH = "data/chunks.db"
INDEX_PATH = "data/faiss_index.bin"
MAPPING_PATH = "data/id_mapping.npy"


class CandidateRetriever:
    def __init__(self):
        self.index = faiss.read_index(INDEX_PATH)
        self.id_mapping = np.load(MAPPING_PATH)
        self.encoder = SentenceTransformer("all-mpnet-base-v2")

    def fetch(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        query_emb = self.encoder.encode([query], normalize_embeddings=True)
        distances, faiss_indices = self.index.search(query_emb, top_k)
        chunk_ids = self.id_mapping[faiss_indices[0]]
        return [(int(cid), float(score)) for cid, score in zip(chunk_ids, distances[0])]


def fetch_chunk_texts(chunk_ids: List[int]) -> Dict[int, str]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    placeholders = ",".join(["?"] * len(chunk_ids))
    cur.execute(f"SELECT chunk_id, chunk_text FROM chunks WHERE chunk_id IN ({placeholders})", tuple(chunk_ids))
    rows = cur.fetchall()
    conn.close()
    return {int(cid): text for cid, text in rows}


def _to_fts_query(raw: str) -> str:
    # Keep only alphanumeric tokens; join with OR to avoid FTS syntax errors
    tokens = re.findall(r"[A-Za-z0-9]+", raw.lower())
    # If empty after cleaning, return a token that won't match anything
    if not tokens:
        return "__nomatch__"
    return " OR ".join(tokens)


def bm25_scores(query: str, chunk_ids: List[int]) -> Dict[int, float]:
    if not chunk_ids:
        return {}
    fts_query = _to_fts_query(query)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    # Use FTS5 BM25 ranking via rank function. We restrict by rowid IN (...)
    placeholders = ",".join(["?"] * len(chunk_ids))
    cur.execute(
        f"""
        SELECT rowid, bm25(chunks_fts) as score
        FROM chunks_fts
        WHERE chunks_fts MATCH ?
        AND rowid IN ({placeholders})
        ORDER BY score
        """,
        tuple([fts_query] + [int(cid) for cid in chunk_ids]),
    )
    rows = cur.fetchall()
    conn.close()
    # FTS bm25: lower is better (since it's a distance). Convert to descending score with negative.
    return {int(rowid): -float(score) for rowid, score in rows}


def compute_features(query: str, top_k: int = 20) -> List[Dict]:
    retriever = CandidateRetriever()
    candidates = retriever.fetch(query, top_k=top_k)
    candidate_ids = [cid for cid, _ in candidates]

    id_to_text = fetch_chunk_texts(candidate_ids)
    id_to_bm25 = bm25_scores(query, candidate_ids)

    # Feature vector per candidate
    features = []
    for cid, vec_score in candidates:
        bm25 = id_to_bm25.get(cid, 0.0)
        text = id_to_text.get(cid, "")
        features.append({
            "chunk_id": cid,
            "vector_score": float(vec_score),
            "bm25_score": float(bm25),
            "text": text,
        })
    # Sort by a simple blend for stable ordering
    features.sort(key=lambda r: (0.7 * r["vector_score"]) + (0.3 * r["bm25_score"]), reverse=True)
    return features


def save_features_for_questions(questions_path: str, out_path: str, top_k: int = 20):
    """
    questions.jsonl lines like: {"q": "...", "positives": [chunk_id,...], "negatives": [chunk_id,...]}
    """
    rows = []
    with open(questions_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            q = item["q"]
            pos = set(item.get("positives", []))
            neg = set(item.get("negatives", []))

            feats = compute_features(q, top_k=top_k)
            # Ensure that labeled chunk_ids are present even if not in top-k
            present_ids = {r["chunk_id"] for r in feats}
            required_ids = (pos | neg) - present_ids
            if required_ids:
                extra_texts = fetch_chunk_texts(list(required_ids))
                extra_bm25 = bm25_scores(q, list(required_ids))
                encoder = CandidateRetriever().encoder
                q_vec = encoder.encode([q], normalize_embeddings=True)[0]
                for cid in required_ids:
                    text = extra_texts.get(int(cid), "")
                    if not text:
                        continue
                    t_vec = encoder.encode([text], normalize_embeddings=True)[0]
                    vec_score = float(np.dot(q_vec, t_vec))
                    feats.append({
                        "chunk_id": int(cid),
                        "vector_score": vec_score,
                        "bm25_score": float(extra_bm25.get(int(cid), 0.0)),
                        "text": text,
                    })
            for r in feats:
                if r["chunk_id"] in pos:
                    label = 1
                elif r["chunk_id"] in neg:
                    label = 0
                else:
                    label = None
                rows.append({
                    "q": q,
                    "chunk_id": r["chunk_id"],
                    "vector_score": r["vector_score"],
                    "bm25_score": r["bm25_score"],
                    "label": label,
                })

    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


