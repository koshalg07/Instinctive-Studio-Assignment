import faiss
import numpy as np
import sqlite3
from sentence_transformers import SentenceTransformer

DB_PATH = "data/chunks.db"
INDEX_PATH = "data/faiss_index.bin"
ID_MAP_PATH = "data/id_mapping.npy"

index = faiss.read_index(INDEX_PATH)

id_mapping = np.load(ID_MAP_PATH)

model = SentenceTransformer("all-mpnet-base-v2")

def search(query, k=5):
    query_emb = model.encode([query], normalize_embeddings=True)

    # Search in FAISS
    scores, indices = index.search(query_emb, k)

    results = []
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    for score, idx in zip(scores[0], indices[0]):
        chunk_id = int(id_mapping[idx])  # map FAISS idx → chunk_id
        cur.execute("SELECT chunk_text, doc_id FROM chunks WHERE chunk_id = ?", (chunk_id,))
        row = cur.fetchone()
        if row:
            chunk_text, doc_id = row
            cur.execute("SELECT title FROM docs WHERE doc_id = ?", (doc_id,))
            doc_title = cur.fetchone()[0]
            results.append({
                "score": float(score),
                "chunk": chunk_text[:300] + "..." if len(chunk_text) > 300 else chunk_text,
                "doc_title": doc_title
            })
    conn.close()
    return results

if __name__ == "__main__":
    query = "What are the key safety requirements for machinery?"
    results = search(query, k=5)
    for r in results:
        print(f"[{r['score']:.4f}] {r['doc_title']} -> {r['chunk']}\n")
