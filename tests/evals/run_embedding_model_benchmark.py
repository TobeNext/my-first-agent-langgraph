# ruff: noqa: E402
"""
Embedding Model Benchmark for Interview Question Retrieval RAG.

Evaluates multiple open-source embedding models on a Chinese/English mixed
interview question retrieval task. For each model, the script:
  1. Loads the model via sentence-transformers (or FlagEmbedding for BGE-M3).
  2. Encodes the question bank as "passage" embeddings.
  3. Encodes each eval-case query as a "query" embedding.
  4. Computes cosine similarity between each query and all questions.
  5. Ranks by similarity and computes standard IR metrics.

Models under evaluation:
  - BAAI/bge-m3              (568M, 1024-dim, dense+sparse, top recommendation)
  - intfloat/multilingual-e5-large   (560M, 1024-dim, retrieval specialist)
  - intfloat/multilingual-e5-base    (278M, 768-dim, balanced speed/accuracy)
  - intfloat/multilingual-e5-small   (118M, 384-dim, CPU-friendly)
  - BAAI/bge-large-zh-v1.5           (326M, 1024-dim, Chinese specialist)

Usage:
  # Pre-download models first (one-time):
  #   pip install huggingface_hub
  #   export HF_ENDPOINT=https://hf-mirror.com   # if behind GFW
  #   python -c "from sentence_transformers import SentenceTransformer; \\
  #     SentenceTransformer('BAAI/bge-m3')"

  # Then run benchmark:
  python -m tests.evals.run_embedding_model_benchmark --offline
  python -m tests.evals.run_embedding_model_benchmark \\
    --models bge-m3,multilingual-e5-base --limit 10
  python -m tests.evals.run_embedding_model_benchmark \\
    --models multilingual-e5-small --output results.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

EMBEDDING_MODELS: dict[str, dict[str, Any]] = {
    "bge-m3": {
        "name": "BAAI/bge-m3",
        "dimension": 1024,
        "description": "BGE-M3: dense+sparse multilingual, retrieval SOTA",
        "query_prefix": "",
        "passage_prefix": "",
        "query_instruction": "Represent this sentence for searching relevant passages: ",
    },
    "multilingual-e5-base": {
        "name": "intfloat/multilingual-e5-base",
        "dimension": 768,
        "description": "E5-base: balanced speed/accuracy",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
    },
    "multilingual-e5-small": {
        "name": "intfloat/multilingual-e5-small",
        "dimension": 384,
        "description": "E5-small: CPU-friendly, fast inference",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
    },
    "bge-large-zh": {
        "name": "BAAI/bge-large-zh-v1.5",
        "dimension": 1024,
        "description": "BGE-large-zh: Chinese specialist",
        "query_prefix": "",
        "passage_prefix": "",
        "query_instruction": "为这个句子生成表示以用于检索相关文章：",
    },
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

BANK_PATH = Path(__file__).resolve().parent / "datasets" / "interview_question_bank.json"
EVAL_CASES_PATH = Path(__file__).resolve().parent / "datasets" / "embedding_eval_cases.json"


def load_question_bank(path: Path = BANK_PATH) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return raw
    return raw["questions"] if isinstance(raw, dict) else []


def load_eval_cases(path: Path = EVAL_CASES_PATH) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw if isinstance(raw, list) else []


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@dataclass
class ModelBenchmarkResult:
    model_key: str
    model_name: str
    dimension: int
    description: str
    load_time_seconds: float
    encode_time_seconds: float
    total_time_seconds: float
    case_results: list[dict[str, Any]] = field(default_factory=list)
    aggregated_metrics: dict[str, float] = field(default_factory=dict)
    error: str | None = None


def cosine_similarity(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between one query vector and all document vectors."""
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    doc_norms = doc_vecs / (np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-10)
    return np.dot(doc_norms, query_norm)


def compute_metrics_for_case(
    query_vec: np.ndarray,
    doc_vecs: np.ndarray,
    question_bank: list[dict[str, Any]],
    case: dict[str, Any],
    *,
    top_k: int = 10,
) -> dict[str, Any]:
    similarities = cosine_similarity(query_vec, doc_vecs)
    ranked_indices = np.argsort(-similarities)
    ranked_ids = [question_bank[int(idx)]["id"] for idx in ranked_indices[:top_k]]
    ranked_scores = [float(similarities[int(idx)]) for idx in ranked_indices[:top_k]]

    expected_ids = set(str(item) for item in case.get("expected_question_ids", []))
    negative_ids = set(str(item) for item in case.get("negative_question_ids", []))

    return {
        "case_id": case["case_id"],
        "ranked_ids": ranked_ids,
        "ranked_scores": [round(score, 4) for score in ranked_scores],
        "recall_at_5": _recall_at_k(ranked_ids[:5], expected_ids),
        "recall_at_10": _recall_at_k(ranked_ids[:10], expected_ids),
        "mrr": _mrr(ranked_ids, expected_ids),
        "ndcg_at_10": _ndcg(ranked_ids[:10], list(expected_ids)),
        "precision_at_5": _precision_at_k(ranked_ids[:5], expected_ids),
        "negative_exclusion": _negative_exclusion(ranked_ids, negative_ids),
        "matched_expected": sorted(expected_ids & set(ranked_ids)),
        "missed_expected": sorted(expected_ids - set(ranked_ids)),
        "leaked_negative": sorted(negative_ids & set(ranked_ids)),
    }


def _recall_at_k(ranked_ids: list[str], expected_ids: set[str]) -> float:
    if not expected_ids:
        return 0.0
    hits = len(expected_ids & set(ranked_ids))
    return round(hits / len(expected_ids), 4)


def _mrr(ranked_ids: list[str], expected_ids: set[str]) -> float:
    for rank, candidate_id in enumerate(ranked_ids, start=1):
        if candidate_id in expected_ids:
            return round(1.0 / rank, 4)
    return 0.0


def _ndcg(ranked_ids: list[str], expected_ids: list[str]) -> float:
    if not expected_ids:
        return 0.0
    expected_set = set(expected_ids)
    dcg = sum(
        (1.0 / math.log2(rank + 2))
        for rank, cid in enumerate(ranked_ids)
        if cid in expected_set
    )
    ideal_hits = min(len(expected_set), len(ranked_ids))
    ideal_dcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_hits))
    return round(dcg / ideal_dcg, 4) if ideal_dcg else 0.0


def _precision_at_k(ranked_ids: list[str], expected_ids: set[str]) -> float:
    if not ranked_ids or not expected_ids:
        return 0.0
    hits = len(expected_ids & set(ranked_ids))
    return round(hits / len(ranked_ids), 4)


def _negative_exclusion(ranked_ids: list[str], negative_ids: set[str]) -> float:
    if not negative_ids:
        return 1.0
    leaked = len(negative_ids & set(ranked_ids))
    return round(1.0 - (leaked / len(negative_ids)), 4)


def aggregate_case_metrics(case_results: list[dict[str, Any]]) -> dict[str, float]:
    metric_names = [
        "recall_at_5",
        "recall_at_10",
        "mrr",
        "ndcg_at_10",
        "precision_at_5",
        "negative_exclusion",
    ]
    result: dict[str, float] = {}
    for name in metric_names:
        values = [float(case[name]) for case in case_results if name in case]
        result[name] = round(sum(values) / len(values), 4) if values else 0.0
    return result


# ---------------------------------------------------------------------------
# Model loading & encoding
# ---------------------------------------------------------------------------


def _has_sentence_transformers() -> bool:
    try:
        import sentence_transformers  # noqa: F401
        return True
    except ImportError:
        return False


def _load_st_model(
    model_definition: dict[str, Any],
    *,
    local_files_only: bool = False,
) -> Any:
    from sentence_transformers import SentenceTransformer

    model_name = model_definition["name"]
    kwargs: dict[str, Any] = {"trust_remote_code": True}

    # Try local cache path first (from direct_download.py)
    local_dir = _local_cache_dir(model_name)
    if local_dir and local_dir.exists():
        return SentenceTransformer(str(local_dir), **kwargs)

    if local_files_only:
        kwargs["local_files_only"] = True
    return SentenceTransformer(model_name, **kwargs)


def _local_cache_dir(model_name: str) -> Path | None:
    """Resolve the local cache dir used by direct_download.py."""
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    dir_name = f"models--{model_name.replace('/', '--')}"
    candidate = cache_root / dir_name
    if candidate.is_dir():
        return candidate
    return None


def encode_questions(
    model: Any,
    question_bank: list[dict[str, Any]],
    model_definition: dict[str, Any],
) -> np.ndarray:
    """Encode all questions as passage embeddings."""
    texts = _build_passage_texts(question_bank, model_definition)
    return _encode_batch(model, texts, model_definition, is_query=False)


def encode_queries(
    model: Any,
    queries: list[str],
    model_definition: dict[str, Any],
) -> np.ndarray:
    """Encode queries with the query prefix/instruction."""
    return _encode_batch(model, queries, model_definition, is_query=True)


def _build_passage_texts(
    question_bank: list[dict[str, Any]],
    model_definition: dict[str, Any],
) -> list[str]:
    texts: list[str] = []
    prefix = model_definition.get("passage_prefix", "")
    for item in question_bank:
        text = item.get("text", "")
        if prefix and not text.startswith(prefix):
            text = prefix + text
        texts.append(text)
    return texts


def _build_query_texts(
    queries: list[str],
    model_definition: dict[str, Any],
) -> list[str]:
    texts: list[str] = []
    query_prefix = model_definition.get("query_prefix", "")
    query_instruction = model_definition.get("query_instruction", "")
    for query in queries:
        text = query
        if query_instruction:
            text = query_instruction + text
        if query_prefix and not text.startswith(query_prefix):
            text = query_prefix + text
        texts.append(text)
    return texts


def _encode_batch(
    model: Any,
    texts: list[str],
    model_definition: dict[str, Any],
    *,
    is_query: bool,
) -> np.ndarray:
    encode_kwargs: dict[str, Any] = {
        "show_progress_bar": False,
        "normalize_embeddings": True,
        "batch_size": 32,
    }
    # BGE-M3 supports `prompt` parameter
    if not model_definition.get("query_prefix") and model_definition.get("query_instruction"):
        if is_query:
            encode_kwargs["prompt"] = model_definition["query_instruction"]
    return np.array(model.encode(texts, **encode_kwargs))


def _download_models(models_arg: str | None) -> int:
    """Pre-download specified models to local HuggingFace cache."""
    print("Pre-downloading embedding models to local cache...")
    selected = (
        [key.strip() for key in models_arg.split(",") if key.strip()]
        if models_arg
        else list(EMBEDDING_MODELS.keys())
    )
    for model_key in selected:
        model_def = EMBEDDING_MODELS.get(model_key)
        if model_def is None:
            print(f"  SKIP: Unknown model key '{model_key}'")
            continue
        print(f"  Downloading {model_def['name']}...")
        try:
            _load_st_model(model_def, local_files_only=False)
            print(f"  ✓ {model_def['name']} downloaded")
        except Exception as exc:
            print(f"  ✗ {model_def['name']} failed: {exc}")
            print("    Tip: set HF_ENDPOINT=https://hf-mirror.com for Chinese mirror")
    return 0


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def run_model_benchmark(
    model_key: str,
    model_definition: dict[str, Any],
    question_bank: list[dict[str, Any]],
    eval_cases: list[dict[str, Any]],
    *,
    top_k: int = 10,
    offline: bool = False,
) -> ModelBenchmarkResult:
    result = ModelBenchmarkResult(
        model_key=model_key,
        model_name=model_definition["name"],
        dimension=model_definition["dimension"],
        description=model_definition["description"],
        load_time_seconds=0.0,
        encode_time_seconds=0.0,
        total_time_seconds=0.0,
    )

    start = time.perf_counter()
    try:
        # Load model
        t0 = time.perf_counter()
        model = _load_st_model(model_definition, local_files_only=offline)
        result.load_time_seconds = round(time.perf_counter() - t0, 2)

        # Encode
        t1 = time.perf_counter()
        doc_vecs = encode_questions(model, question_bank, model_definition)
        queries_raw = [case["query"] for case in eval_cases]
        query_texts = _build_query_texts(queries_raw, model_definition)
        query_vecs = encode_queries(model, query_texts, model_definition)
        result.encode_time_seconds = round(time.perf_counter() - t1, 2)

        # Evaluate each case
        for idx, case in enumerate(eval_cases):
            case_result = compute_metrics_for_case(
                query_vecs[idx],
                doc_vecs,
                question_bank,
                case,
                top_k=top_k,
            )
            result.case_results.append(case_result)

        result.aggregated_metrics = aggregate_case_metrics(result.case_results)
    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"
    finally:
        result.total_time_seconds = round(time.perf_counter() - start, 2)

    return result


def run_full_benchmark(
    models: list[str] | None = None,
    *,
    top_k: int = 10,
    limit_cases: int | None = None,
    offline: bool = False,
) -> list[ModelBenchmarkResult]:
    if not _has_sentence_transformers():
        print(
            "ERROR: sentence-transformers is not installed.\n"
            "Install with: pip install sentence-transformers",
            file=sys.stderr,
        )
        return []

    question_bank = load_question_bank()
    eval_cases = load_eval_cases()
    if limit_cases is not None:
        eval_cases = eval_cases[:limit_cases]

    print(f"Question bank: {len(question_bank)} questions")
    print(f"Eval cases: {len(eval_cases)} queries")
    print(f"Top-K: {top_k}")
    print()

    selected = models or list(EMBEDDING_MODELS.keys())
    results: list[ModelBenchmarkResult] = []

    for model_key in selected:
        model_def = EMBEDDING_MODELS.get(model_key)
        if model_def is None:
            print(f"SKIP: Unknown model key '{model_key}'")
            continue
        print(f"Evaluating: {model_def['name']} ({model_def['description']}) ...")
        result = run_model_benchmark(
            model_key,
            model_def,
            question_bank,
            eval_cases,
            top_k=top_k,
            offline=offline,
        )
        results.append(result)
        if result.error:
            print(f"  ERROR: {result.error}")
        else:
            print(f"  Load: {result.load_time_seconds}s  Encode: {result.encode_time_seconds}s  "
                  f"Total: {result.total_time_seconds}s")
            metrics = result.aggregated_metrics
            print(f"  Recall@5={metrics.get('recall_at_5', 'N/A')}  "
                  f"Recall@10={metrics.get('recall_at_10', 'N/A')}  "
                  f"MRR={metrics.get('mrr', 'N/A')}  "
                  f"NDCG@10={metrics.get('ndcg_at_10', 'N/A')}")
            print(f"  Precision@5={metrics.get('precision_at_5', 'N/A')}  "
                  f"NegExclusion={metrics.get('negative_exclusion', 'N/A')}")
        print()

    return results


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def _model_rank_label(rank: int) -> str:
    labels = {0: "🥇", 1: "🥈", 2: "🥉"}
    return labels.get(rank, f"#{rank + 1}")


def print_comparison_table(results: list[ModelBenchmarkResult]) -> None:
    valid = [r for r in results if not r.error]
    if not valid:
        print("No successful model results to compare.")
        return

    # Rank models by Recall@10 (primary metric)
    ranked = sorted(
        valid,
        key=lambda r: r.aggregated_metrics.get("recall_at_10", 0),
        reverse=True,
    )

    header = (
        f"{'Rank':<5} {'Model':<28} {'Dim':<5} {'Load(s)':<8} "
        f"{'Enc(s)':<8} {'R@5':<7} {'R@10':<7} {'MRR':<7} "
        f"{'NDCG':<7} {'P@5':<7} {'NegEx':<7}"
    )
    sep = "-" * len(header)

    print("\n" + "=" * len(header))
    print("  EMBEDDING MODEL BENCHMARK — Interview Question Retrieval RAG")
    print("=" * len(header))
    print(header)
    print(sep)

    for rank, r in enumerate(ranked):
        m = r.aggregated_metrics
        print(
            f"{_model_rank_label(rank):<5} {r.model_key:<28} {r.dimension:<5} "
            f"{r.load_time_seconds:<8.1f} {r.encode_time_seconds:<8.1f} "
            f"{m.get('recall_at_5', 0):<7.4f} {m.get('recall_at_10', 0):<7.4f} "
            f"{m.get('mrr', 0):<7.4f} {m.get('ndcg_at_10', 0):<7.4f} "
            f"{m.get('precision_at_5', 0):<7.4f} {m.get('negative_exclusion', 0):<7.4f}"
        )

    print(sep)

    # Best model recommendation
    best = ranked[0]
    print(f"\n✅ 推荐: {best.model_name}")
    print(f"   理由: Recall@10={best.aggregated_metrics.get('recall_at_10', 0):.4f}, "
          f"MRR={best.aggregated_metrics.get('mrr', 0):.4f}")
    print()

    # Per-case detail for the best model
    print(f"--- 最佳模型逐 Case 详情 ({best.model_key}) ---")
    for case_result in best.case_results:
        matched = case_result.get("matched_expected", [])
        missed = case_result.get("missed_expected", [])
        leaked = case_result.get("leaked_negative", [])
        print(f"  [{case_result['case_id']}]")
        print(f"    R@10={case_result['recall_at_10']}  MRR={case_result['mrr']}  "
              f"NDCG={case_result['ndcg_at_10']}")
        if matched:
            print(f"    ✅ Matched: {', '.join(matched)}")
        if missed:
            print(f"    ❌ Missed:  {', '.join(missed)}")
        if leaked:
            print(f"    ⚠️ Leaked:  {', '.join(leaked)}")


def save_results_json(
    results: list[ModelBenchmarkResult],
    path: Path,
    question_bank_size: int,
    eval_cases_count: int,
    top_k: int,
) -> None:
    output = {
        "config": {
            "question_bank_size": question_bank_size,
            "eval_cases_count": eval_cases_count,
            "top_k": top_k,
        },
        "models": [
            {
                "model_key": r.model_key,
                "model_name": r.model_name,
                "dimension": r.dimension,
                "description": r.description,
                "load_time_seconds": r.load_time_seconds,
                "encode_time_seconds": r.encode_time_seconds,
                "total_time_seconds": r.total_time_seconds,
                "aggregated_metrics": r.aggregated_metrics,
                "case_results": r.case_results,
                "error": r.error,
            }
            for r in results
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Results saved to: {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embedding Model Benchmark for Interview Question Retrieval RAG",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated model keys to evaluate (default: all). "
        "Available: bge-m3, multilingual-e5-large, multilingual-e5-base, "
        "multilingual-e5-small, bge-large-zh",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Top-K for retrieval metrics")
    parser.add_argument("--limit", type=int, default=None, help="Limit eval cases")
    parser.add_argument("--output", type=Path, default=None, help="Save JSON results to path")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Only use locally cached models (skip HuggingFace download)",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Pre-download all models to local cache then exit",
    )
    parser.add_argument(
        "--model-list",
        action="store_true",
        help="Print available models and exit",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    if args.model_list:
        print("Available embedding models:")
        for key, info in EMBEDDING_MODELS.items():
            print(f"  {key:<28} {info['name']:<40} {info['dimension']}d  {info['description']}")
        return 0

    if args.download:
        return _download_models(args.models)

    model_keys: list[str] | None = None
    if args.models:
        model_keys = [key.strip() for key in args.models.split(",") if key.strip()]

    question_bank = load_question_bank()
    eval_cases = load_eval_cases()
    if args.limit:
        eval_cases = eval_cases[: args.limit]

    results = run_full_benchmark(
        models=model_keys,
        top_k=args.top_k,
        limit_cases=args.limit,
        offline=args.offline,
    )

    if not results:
        return 1

    print_comparison_table(results)

    if args.output:
        save_results_json(
            results,
            args.output,
            len(question_bank),
            len(eval_cases),
            args.top_k,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
