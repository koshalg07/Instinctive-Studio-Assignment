import json
import csv
from typing import List, Dict, Any

from rerank import fetch_candidates_faiss as fetch_candidates
from rerank import rerank as rerank_candidates
from features import _to_fts_query  


def run_mode(q: str, k: int, mode: str):
    candidates = fetch_candidates(q, top_k=max(k, 50))
    if mode == "baseline":
        ranked = [
            {"chunk_id": int(cid), "score": float(score), "text": text}
            for cid, score, text in sorted(candidates, key=lambda r: r[1], reverse=True)[:k]
        ]
        top_score = ranked[0]["score"] if ranked else 0.0
        return ranked, top_score
    else:
        reranked = rerank_candidates(q, candidates)[:k]
        top_score = reranked[0].get("rerank_score", 0.0) if reranked else 0.0
        return reranked, top_score


def evaluate(questions_path: str, out_csv: str, k: int = 5):
    rows = []
    with open(questions_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            q = item["q"]

            base_ranked, base_top = run_mode(q, k, mode="baseline")
            rr_ranked, rr_top = run_mode(q, k, mode="rerank")

            abstain_base = base_top < 0.30
            abstain_rr = rr_top < 0.45

            rows.append({
                "question": q,
                "baseline_top": f"{base_top:.3f}",
                "baseline_abstain": abstain_base,
                "rerank_top": f"{rr_top:.3f}",
                "rerank_abstain": abstain_rr,
            })

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"Wrote results to {out_csv}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("questions", help="questions.jsonl path")
    parser.add_argument("--out", default="data/eval_results.csv")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    evaluate(args.questions, args.out, k=args.k)


