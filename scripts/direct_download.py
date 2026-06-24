"""
Direct model downloader using requests with progress bars.
Downloads model files from HF mirror and stores in HF cache format.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

MIRROR = "https://hf-mirror.com"
CACHE = Path.home() / ".cache" / "huggingface" / "hub"


def download_file(
    url: str,
    dest: Path,
    *,
    timeout: int = 30,
    max_retries: int = 5,
) -> bool:
    """Download a single file with resume + retry on mid-stream failures."""
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, max_retries + 1):
        existing = dest.stat().st_size if dest.exists() else 0
        headers = {"Range": f"bytes={existing}-"} if existing > 0 else {}

        try:
            resp = requests.get(url, headers=headers, stream=True, timeout=(30, timeout))
        except requests.RequestException as exc:
            if attempt < max_retries:
                print(f"\n    Retry {attempt}/{max_retries}: {exc}")
                time.sleep(2)
                continue
            print(f"\n    FAILED after {max_retries} retries: {exc}")
            return False

        if resp.status_code == 416:
            print(f"    Already complete ({existing/1e6:.1f} MB)")
            return True

        if resp.status_code not in (200, 206):
            print(f"\n    HTTP {resp.status_code}")
            if attempt < max_retries:
                time.sleep(2)
                continue
            return False

        total = int(resp.headers.get("content-length", 0)) + existing
        mode = "ab" if existing > 0 else "wb"
        chunk_size = 8 * 1024 * 1024
        downloaded = existing
        t0 = time.perf_counter()

        try:
            with open(dest, mode) as f:
                for chunk in resp.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        elapsed = time.perf_counter() - t0
                        speed = downloaded / elapsed / 1e6 if elapsed > 0 else 0
                        pct = downloaded / total * 100 if total > 0 else 0
                        print(
                            f"\r    {downloaded/1e6:.1f}/{total/1e6:.1f} MB "
                            f"({pct:.0f}%) {speed:.1f} MB/s   ",
                            end="",
                        )
            print()
            return True
        except (requests.RequestException, OSError) as exc:
            if attempt < max_retries:
                print(f"\n    Retry {attempt}/{max_retries} (resume @ {downloaded/1e6:.1f}MB): {exc}")
                time.sleep(2)
                continue
            print(f"\n    FAILED after {max_retries} retries: {exc}")
            return False

    return False


def download_model(repo_id: str, label: str) -> bool:
    """Download all required files for a SentenceTransformer model."""
    print(f"[{label}] {repo_id}")

    # Step 1: Get file list from the repo
    import requests

    api_url = f"{MIRROR}/api/models/{repo_id}"
    try:
        resp = requests.get(api_url, timeout=30)
        if resp.status_code != 200:
            print(f"  API error: {resp.status_code}")
            return False
        siblings = resp.json().get("siblings", [])
    except Exception as exc:
        print(f"  API error: {exc}")
        return False

    # Step 2: Filter to required files (skip ONNX, TF, Rust, flax)
    skip_patterns = (
        ".onnx", "onnx/", "tf_model", "rust_model", "flax_model",
        ".h5", ".msgpack", ".eval_results/", ".cache/", "openvino/",
    )
    needed = [
        s["rfilename"]
        for s in siblings
        if not any(p in s["rfilename"] for p in skip_patterns)
    ]

    # Step 3: Download each file
    cache_dir = CACHE / f"models--{repo_id.replace('/', '--')}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    for filename in needed:
        url = f"{MIRROR}/{repo_id}/resolve/main/{filename}"
        dest = cache_dir / filename
        print(f"  {filename} ...")
        if not download_file(url, dest):
            print(f"  FAILED: {filename}")
            return False

    # Step 4: Verify with SentenceTransformer
    print(f"  Verifying with SentenceTransformer...")
    try:
        from sentence_transformers import SentenceTransformer

        m = SentenceTransformer(str(cache_dir), trust_remote_code=True)
        dim = m.get_sentence_embedding_dimension()
        print(f"  VERIFIED: dim={dim}")
    except Exception as exc:
        print(f"  Load error (may still work): {exc}")

    return True


def main() -> int:
    models = [
        ("intfloat/multilingual-e5-small", "E5-small"),
        ("intfloat/multilingual-e5-base", "E5-base"),
        ("BAAI/bge-large-zh-v1.5", "BGE-large-zh"),
        ("BAAI/bge-m3", "BGE-M3"),
    ]

    print(f"Mirror: {MIRROR}")
    print(f"Cache: {CACHE}")
    print()

    failed = []
    for repo_id, label in models:
        ok = download_model(repo_id, label)
        if not ok:
            failed.append(repo_id)
        print()

    if failed:
        print(f"FAILED: {failed}")
        return 1
    print("All models downloaded!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
