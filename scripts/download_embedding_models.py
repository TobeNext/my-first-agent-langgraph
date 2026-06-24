"""
Quick helper to pre-download embedding models before running the benchmark.

Usage (Windows PowerShell):
  $env:HF_ENDPOINT="https://hf-mirror.com"
  python scripts/download_embedding_models.py

Usage (Linux/macOS):
  HF_ENDPOINT=https://hf-mirror.com python scripts/download_embedding_models.py

Downloads all 5 models to the local HuggingFace cache (~8 GB total).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tests.evals.run_embedding_model_benchmark import EMBEDDING_MODELS, _load_st_model  # noqa: E402


def main() -> int:
    print("Pre-downloading embedding models to local HuggingFace cache...")
    print(f"HF_ENDPOINT={__import__('os').environ.get('HF_ENDPOINT', '(default)')}")
    print()

    failed: list[str] = []
    for model_key, model_def in EMBEDDING_MODELS.items():
        print(f"[{model_key}] Downloading {model_def['name']} ({model_def['description']})...")
        try:
            _load_st_model(model_def, local_files_only=False)
            print("  ✓ Done")
        except Exception as exc:
            print(f"  ✗ Failed: {exc}")
            failed.append(model_key)
        print()

    if failed:
        print(f"Failed models: {', '.join(failed)}")
        print("Tip: set HF_ENDPOINT=https://hf-mirror.com to use the Chinese mirror.")
        return 1
    print("All models downloaded successfully! Run benchmark with --offline flag.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
