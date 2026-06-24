"""
Download models one by one using SentenceTransformer (proven approach).
Minimal — just loads each model to trigger cache download.
"""

from __future__ import annotations

import os
import sys
import time

MODELS = [
    ("intfloat/multilingual-e5-small", "E5-small"),
    ("intfloat/multilingual-e5-base", "E5-base"),
    ("intfloat/multilingual-e5-large", "E5-large"),
    ("BAAI/bge-large-zh-v1.5", "BGE-large-zh"),
    ("BAAI/bge-m3", "BGE-M3"),
]

def main() -> int:
    print(f"HF_ENDPOINT={os.environ.get('HF_ENDPOINT', '(default)')}")
    print()

    from sentence_transformers import SentenceTransformer

    failed = []
    for repo_id, label in MODELS:
        print(f"[{label}] {repo_id}")
        t0 = time.perf_counter()
        try:
            m = SentenceTransformer(repo_id, trust_remote_code=True)
            dim = m.get_sentence_embedding_dimension()
            elapsed = time.perf_counter() - t0
            print(f"  OK: dim={dim} ({elapsed:.1f}s)")
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            print(f"  FAIL ({elapsed:.1f}s): {exc}")
            failed.append(repo_id)
        print()

    if failed:
        print(f"FAILED: {failed}")
        return 1
    print("All 5 models downloaded & verified!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
