import sqlite3
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

DB_PATH = "data/chunks.db"
INDEX_PATH = "data/faiss_index.bin"
MAPPING_PATH = "data/id_mapping.npy"

# Connect to sqlite
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()
cur.execute("SELECT chunk_id, chunk_text FROM chunks")
rows = cur.fetchall()
conn.close()

chunk_ids = [r[0] for r in rows]
texts = [r[1] for r in rows]
print(f"Loaded {len(texts)} chunks")


# Generate embeddings

model = SentenceTransformer("all-mpnet-base-v2")
embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
embeddings = np.array(embeddings).astype("float32")  
print("Embeddings generated:", embeddings.shape)

# Build FAISS index

dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)  # cosine similarity via inner product after normalization
faiss.normalize_L2(embeddings)
index.add(embeddings)
print("FAISS index built with", index.ntotal, "vectors")

# Save index and mapping

faiss.write_index(index, INDEX_PATH)
np.save(MAPPING_PATH, np.array(chunk_ids))
print("Index saved to", INDEX_PATH)
print("ID mapping saved to", MAPPING_PATH)
