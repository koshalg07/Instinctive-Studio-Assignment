import os
import re
import sqlite3
from typing import List, Dict, Any

from flask import Flask, request, jsonify

from rerank import fetch_candidates_faiss as fetch_candidates
from rerank import rerank as rerank_candidates

DB_PATH = "data/chunks.db"

app = Flask(__name__)


def get_doc_meta(chunk_ids: List[int]) -> Dict[int, Dict[str, Any]]:
    if not chunk_ids:
        return {}
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    placeholders = ",".join(["?"] * len(chunk_ids))
    cur.execute(
        f"""
        SELECT c.chunk_id, d.title, d.url, d.doc_id
        FROM chunks c
        JOIN docs d ON c.doc_id = d.doc_id
        WHERE c.chunk_id IN ({placeholders})
        """,
        tuple(chunk_ids),
    )
    rows = cur.fetchall()
    conn.close()
    return {int(cid): {"title": title, "url": url} for cid, title, url, _ in rows}


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def best_extractive_snippet(query: str, text: str, max_sents: int = 2) -> str:
    q_terms = set(re.findall(r"[a-z0-9]+", query.lower()))
    sents = split_sentences(text)
    scored = []
    for s in sents:
        s_terms = set(re.findall(r"[a-z0-9]+", s.lower()))
        overlap = len(q_terms & s_terms)
        scored.append((overlap, s))
    scored.sort(key=lambda x: (x[0], len(x[1])), reverse=True)
    top = [s for _, s in scored[:max_sents]]
    return " " .join(top)[:500]


def retrieve(query: str, k: int, mode: str) -> List[Dict[str, Any]]:
    candidates = fetch_candidates(query, top_k=max(k, 50))
    # candidates: list of (chunk_id, base_score, text)
    if mode == "rerank":
        reranked = rerank_candidates(query, candidates)
        top = reranked[:k]
        chunk_ids = [r["chunk_id"] for r in top]
        meta = get_doc_meta(chunk_ids)
        contexts = []
        for r in top:
            info = meta.get(r["chunk_id"], {"title": None, "url": None})
            contexts.append({
                "chunk_id": r["chunk_id"],
                "score": r.get("base_score"),
                "rerank_score": r.get("rerank_score"),
                "bm25_score": r.get("bm25_score"),
                "title": info["title"],
                "url": info["url"],
                "text": r["text"],
            })
        return contexts
    else:
        # baseline: sort by base_score
        candidates_sorted = sorted(
            (
                {"chunk_id": int(cid), "base_score": float(score), "text": txt}
                for cid, score, txt in candidates
            ),
            key=lambda r: r["base_score"],
            reverse=True,
        )[:k]
        chunk_ids = [r["chunk_id"] for r in candidates_sorted]
        meta = get_doc_meta(chunk_ids)
        contexts = []
        for r in candidates_sorted:
            info = meta.get(r["chunk_id"], {"title": None, "url": None})
            contexts.append({
                "chunk_id": r["chunk_id"],
                "score": r["base_score"],
                "title": info["title"],
                "url": info["url"],
                "text": r["text"][:300] + ("..." if len(r["text"]) > 300 else ""),
            })
        return contexts


def build_answer(query: str, contexts: List[Dict[str, Any]], mode: str):
    if not contexts:
        return None, "no_contexts"
    # Abstain thresholds
    if mode == "rerank":
        top_score = contexts[0].get("rerank_score", 0.0) or 0.0
        if top_score < 0.45:
            return None, f"low_confidence_rerank:{top_score:.3f}"
    else:
        top_score = contexts[0].get("score", 0.0) or 0.0
        if top_score < 0.30:
            return None, f"low_confidence_baseline:{top_score:.3f}"

    top = contexts[0]
    snippet = best_extractive_snippet(query, top["text"]) or top["text"][:300]
    citation = {
        "title": top.get("title"),
        "url": top.get("url"),
        "chunk_id": top.get("chunk_id"),
    }
    answer = f"{snippet}"
    return {"text": answer, "citation": citation}, None


@app.post("/ask")
def ask():
    data = request.get_json(force=True) or {}
    q = data.get("q", "").strip()
    k = int(data.get("k", 5))
    mode = data.get("mode", "rerank")  # "baseline" | "rerank"
    if not q:
        return jsonify({"error": "missing q"}), 400

    contexts = retrieve(q, k=k, mode=mode)
    answer, abstain_reason = build_answer(q, contexts, mode)
    return jsonify({
        "answer": answer,  # or null
        "contexts": contexts,
        "reranker_used": mode == "rerank",
        "abstain_reason": abstain_reason,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)


