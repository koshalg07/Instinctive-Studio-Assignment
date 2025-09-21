import re
import hashlib

def chunk_text(text, max_chars=1200, overlap=200):
    """Split text into overlapping chunks at sentence boundaries."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, current = [], []

    for sent in sentences:
        if sum(len(s) for s in current) + len(sent) < max_chars:
            current.append(sent)
        else:
            chunk = " ".join(current).strip()
            if chunk:
                chunks.append(chunk)
            # Start new chunk with overlap
            current = current[-overlap:] if overlap and current else []
            current.append(sent)

    if current:
        chunks.append(" ".join(current).strip())
    return chunks

def sha1_hash(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()
