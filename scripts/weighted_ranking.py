"""
Weighted scoring for Interview Agent Round-1 RAG quality.
Formula: Score = 0.40×R@5 + 0.30×MRR + 0.20×P@5 + 0.10×SpeedBonus
"""

import json
from pathlib import Path

RESULTS_PATH = Path(__file__).resolve().parents[1] / "EmbeddingBenchmark" / "benchmark_results_10k.json"

data = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
models = data["models"]

results = []
for m in models:
    mm = m["aggregated_metrics"]
    results.append(
        {
            "model": m["model_key"],
            "name": m["model_name"],
            "dim": m["dimension"],
            "R5": mm["recall_at_5"],
            "R10": mm["recall_at_10"],
            "MRR": mm["mrr"],
            "P5": mm["precision_at_5"],
            "enc_s": m["encode_time_seconds"],
            "load_s": m["load_time_seconds"],
        }
    )

# Normalize encode time -> speed bonus (faster = higher)
enc_times = [r["enc_s"] for r in results]
min_enc, max_enc = min(enc_times), max(enc_times)
for r in results:
    if max_enc == min_enc:
        r["speed"] = 1.0
    else:
        r["speed"] = 1.0 - (r["enc_s"] - min_enc) / (max_enc - min_enc)

# Weighted score for Interview Agent Round-1 RAG
# Recall@5: 40% — first 5 results must include the right questions
# MRR:      30% — best answer should be ranked #1
# P@5:      20% — all top 5 should be relevant
# Speed:    10% — welcome but not critical in first round
for r in results:
    r["score"] = 0.40 * r["R5"] + 0.30 * r["MRR"] + 0.20 * r["P5"] + 0.10 * r["speed"]

results.sort(key=lambda r: r["score"], reverse=True)

# --- Output ---
hdr = f"{'Rank':<5} {'Model':<26} {'Score':<7} {'R@5':<7} {'MRR':<7} {'P@5':<7} {'Enc(s)':<8} {'Speed':<7}"
sep = "-" * len(hdr)

print("=" * len(hdr))
print("  INTERVIEW AGENT ROUND-1 RAG — Weighted Model Ranking")
print("=" * len(hdr))
print(hdr)
print(sep)

for i, r in enumerate(results):
    label = ["🥇 1st", "🥈 2nd", "🥉 3rd"][i] if i < 3 else f"   #{i + 1} "
    print(
        f"{label:<5} {r['model']:<26} {r['score']:<7.4f} "
        f"{r['R5']:<7.4f} {r['MRR']:<7.4f} {r['P5']:<7.4f} "
        f"{r['enc_s']:<8.1f} {r['speed']:<7.4f}"
    )

print(sep)
print()

best = results[0]
print(f"✅ 最终推荐: {best['name']}  (dim={best['dim']})")
print(f"   加权得分: {best['score']:.4f}")
print(f"   Recall@5={best['R5']:.4f}  MRR={best['MRR']:.4f}  Precision@5={best['P5']:.4f}")
print(f"   编码耗时: {best['enc_s']:.1f}s")
print()

# Detail table
print("--- 各指标详细对比 ---")
col_w = 26
print(f"{'指标':<12} {'权重':<7} " + "".join(f"{r['model']:<{col_w}}" for r in results))
print("-" * (12 + 7 + col_w * len(results)))

rows = [
    ("Recall@5", "40%", "R5"),
    ("MRR", "30%", "MRR"),
    ("Precision@5", "20%", "P5"),
    ("Speed Bonus", "10%", "speed"),
    ("加权总分", "100%", "score"),
]
for metric, weight, key in rows:
    print(f"{metric:<12} {weight:<7} " + "".join(f"{r[key]:<{col_w}.4f}" for r in results))

print()
print("Score = 0.40 × Recall@5 + 0.30 × MRR + 0.20 × Precision@5 + 0.10 × SpeedBonus")
print()
print("SpeedBonus = 1.0 - (encode_time - min_time) / (max_time - min_time)")
