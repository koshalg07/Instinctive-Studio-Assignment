import os
import json
import sqlite3
import pdfplumber
from pathlib import Path
from utils import chunk_text, sha1_hash

DB_PATH = "data/chunks.db"
PDF_DIR = "data/pdfs"
SOURCES_FILE = "sources copy.json"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS chunks;")
    cur.execute("DROP TABLE IF EXISTS docs;")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS docs (
            doc_id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            title TEXT,
            url TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id INTEGER,
            chunk_text TEXT,
            chunk_sha1 TEXT,
            FOREIGN KEY (doc_id) REFERENCES docs(doc_id)
        )
    """)
    conn.commit()
    return conn

def extract_text_from_pdf(path):
    text = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
    return "\n".join(text)

def main():
    conn = init_db()
    cur = conn.cursor()

    with open(SOURCES_FILE, "r") as f:
        sources = json.load(f)

    for src in sources:
        filename, title, url = src["filename"], src["title"], src["url"]
        pdf_path = os.path.join(PDF_DIR, filename)

        if not os.path.exists(pdf_path):
            print(f"⚠️ Missing {pdf_path}, skipping")
            continue

        print(f"📄 Processing {filename} ...")
        full_text = extract_text_from_pdf(pdf_path)

        if not full_text.strip():
            print(f"⚠️ No text extracted from {filename}, skipping")
            continue

        # Insert doc row
        cur.execute("INSERT INTO docs (filename, title, url) VALUES (?, ?, ?)",
                    (filename, title, url))
        doc_id = cur.lastrowid

        # Chunk text
        chunks = chunk_text(full_text, max_chars=1200, overlap=200)
        for ch in chunks:
            cur.execute("INSERT INTO chunks (doc_id, chunk_text, chunk_sha1) VALUES (?, ?, ?)",
                        (doc_id, ch, sha1_hash(ch)))

        conn.commit()
    conn.close()
    print("✅ Ingestion complete!")

if __name__ == "__main__":
    main()
