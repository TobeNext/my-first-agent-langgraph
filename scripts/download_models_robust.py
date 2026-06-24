"""
Robust model downloader — downloads all 5 embedding models from HF mirror.
Uses huggingface_hub.snapshot_download for reliable downloads with progress.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

def main() -> int:
    # Models ordered smallest to largest
    models: list[tuple[str, str]] = [
        ("intfloat/multilingual-e5-small", "E5-small (118M, ~470MB)"),
        ("BAAI/bge-large-zh-v1.5", "BGE-large-zh (326M, ~1.3GB)"),
        ("intfloat/multilingual-e5-base", "E5-base (278M, ~1.1GB)"),
        ("intfloat/multilingual-e5-large", "E5-large (560M, ~2.2GB)"),
        ("BAAI/bge-m3", "BGE-M3 (568M, ~2.2GB)"),
    ]

    print("HF_ENDPOINT:", os.environ.get("HF_ENDPOINT", "(default)"))
    print()

    from huggingface_hub import snapshot_download

    failed: list[str] = []

    for repo_id, description in models:
        print(f"[{repo_id}] {description}")
        print(f"  Starting download...")
        t0 = time.perf_counter()
        try:
            path = snapshot_download(
                repo_id,
                resume_download=True,
                max_workers=4,
                ignore_patterns=["*.onnx*", "onnx/**", "rust_model.ot", "tf_model.h5"],
            )
            elapsed = time.perf_counter() - t0
            print(f"  OK ({elapsed:.1f}s) -> {path}")

            # Verify by loading with SentenceTransformer
            print(f"  Verifying model load...")
            t1 = time.perf_counter()
            from sentence_transformers import SentenceTransformer
            m = SentenceTransformer(repo_id, local_files_only=True, trust_remote_code=True)
            dim = m.get_sentence_embedding_dimension()
            print(f"  Verified: dim={dim} ({time.perf_counter() - t1:.1f}s)")
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            print(f"  FAILED after {elapsed:.1f}s: {exc}")
            failed.append(repo_id)
        print()

    if failed:
        print(f"FAILED models: {failed}")
        return 1
    print("All 5 models downloaded and verified!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
