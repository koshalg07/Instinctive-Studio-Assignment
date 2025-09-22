import os
import json
import subprocess
from pathlib import Path

import requests

DATA_DIR = Path("data")
PDF_DIR = DATA_DIR / "pdfs"
SOURCES_PATH = Path("sources copy.json")


def ensure_dirs():
    DATA_DIR.mkdir(exist_ok=True)
    PDF_DIR.mkdir(parents=True, exist_ok=True)


def download_pdfs():
    with open(SOURCES_PATH, "r", encoding="utf-8") as f:
        sources = json.load(f)
    for src in sources:
        filename, url = src["filename"], src["url"]
        out_path = PDF_DIR / filename
        if out_path.exists() and out_path.stat().st_size > 0:
            print(f"✓ Exists: {filename}")
            continue
        print(f"↓ Downloading: {filename} from {url}")
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(out_path, "wb") as f_out:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f_out.write(chunk)
        except Exception as e:
            print(f"⚠️  Failed to download {url}: {e}")


def run_cmd(cmd: list[str]):
    print("$", " ".join(cmd))
    subprocess.check_call(cmd)


def main():
    ensure_dirs()
    download_pdfs()
    # Build DB and index
    run_cmd(["python", "ingest.py"]) 
    run_cmd(["python", "build_index.py"]) 
    print("✅ Bootstrap complete. Start the API with: python api.py")


if __name__ == "__main__":
    main()


