# Embedding Model Benchmark — Full Pipeline
# Downloads all 5 models, runs benchmark, saves results
# Usage: .\scripts\run_full_benchmark.ps1

$ErrorActionPreference = "Continue"
$ProgressPreference = "SilentlyContinue"

# --- Config ---
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$PYTHON = "C:/Python314/python.exe"
$BENCHMARK_DIR = "G:\project\my-first-agent\my-first-agent-langgraph\EmbeddingBenchmark"
$LOG_FILE = Join-Path $BENCHMARK_DIR "benchmark_run.log"

# Create output dir
New-Item -ItemType Directory -Force -Path $BENCHMARK_DIR | Out-Null

# --- Logging ---
function Write-Log {
    param([string]$Message)
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$ts] $Message"
    Write-Host $line
    Add-Content -Path $LOG_FILE -Value $line
}

# --- Step 1: Download models ---
Write-Log "========== STEP 1: Download Models =========="
$models = @(
    @{Key="multilingual-e5-small";  Name="intfloat/multilingual-e5-small"},
    @{Key="multilingual-e5-base";   Name="intfloat/multilingual-e5-base"},
    @{Key="multilingual-e5-large";  Name="intfloat/multilingual-e5-large"},
    @{Key="bge-large-zh";           Name="BAAI/bge-large-zh-v1.5"},
    @{Key="bge-m3";                 Name="BAAI/bge-m3"}
)

foreach ($m in $models) {
    Write-Log "Downloading: $($m.Name) ..."
    $code = @"
from sentence_transformers import SentenceTransformer
m = SentenceTransformer('$($m.Name)')
print('OK dim=' + str(m.get_sentence_embedding_dimension()))
"@
    $result = & $PYTHON -c $code 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Log "  OK: $($m.Name)"
    } else {
        Write-Log "  FAIL: $($m.Name) — $result"
    }
}

# --- Step 2: Run benchmark ---
Write-Log "========== STEP 2: Run Benchmark =========="
$outputJson = Join-Path $BENCHMARK_DIR "benchmark_results.json"

Write-Log "Running benchmark (all 5 models, 30 eval cases, top-k=10)..."
& $PYTHON -m tests.evals.run_embedding_model_benchmark `
    --offline `
    --top-k 10 `
    --output $outputJson 2>&1 | ForEach-Object {
        Write-Log $_
    }

if ($LASTEXITCODE -eq 0) {
    Write-Log "Benchmark completed successfully."
} else {
    Write-Log "Benchmark exited with code $LASTEXITCODE"
}

# --- Step 3: Summary ---
Write-Log "========== DONE =========="
Write-Log "Results saved to: $outputJson"
Write-Log "Log saved to: $LOG_FILE"

if (Test-Path $outputJson) {
    $summaryScript = Join-Path $BENCHMARK_DIR "_print_summary.py"
@"
import json, sys
results_path = r"$outputJson"
data = json.load(open(results_path, "r", encoding="utf-8"))
models = sorted(data["models"], key=lambda m: m["aggregated_metrics"].get("recall_at_10", 0), reverse=True)
header = f"{'Rank':<5} {'Model':<28} {'R@5':<7} {'R@10':<7} {'MRR':<7} {'NDCG':<7} {'P@5':<7} {'NegEx':<7}"
print(header)
print("-" * len(header))
for i, m in enumerate(models):
    mm = m["aggregated_metrics"]
    label = ["1st", "2nd", "3rd"][i] if i < 3 else f"#{i+1} "
    line = (
        f"{label:<5} {m['model_key']:<28} "
        f"{mm.get('recall_at_5',0):<7.4f} {mm.get('recall_at_10',0):<7.4f} "
        f"{mm.get('mrr',0):<7.4f} {mm.get('ndcg_at_10',0):<7.4f} "
        f"{mm.get('precision_at_5',0):<7.4f} {mm.get('negative_exclusion',0):<7.4f}"
    )
    print(line)
if models:
    best = models[0]
    print()
    print(f"BEST: {best['model_name']}")
    print(f"  Recall@10 = {best['aggregated_metrics'].get('recall_at_10', 0):.4f}")
    print(f"  MRR       = {best['aggregated_metrics'].get('mrr', 0):.4f}")
    print(f"  NDCG@10   = {best['aggregated_metrics'].get('ndcg_at_10', 0):.4f}")
    print(f"  Dims      = {best['dimension']}")
    print(f"  Load time = {best['load_time_seconds']}s")
    print(f"  Encode    = {best['encode_time_seconds']}s")
"@ | Set-Content -Path $summaryScript -Encoding UTF8

    Write-Log ""
    Write-Log "--- Quick Summary ---"
    & $PYTHON $summaryScript 2>&1 | ForEach-Object { Write-Log $_ }
}

Write-Host ""
Write-Host "Full log: $LOG_FILE"
Write-Host "Results JSON: $outputJson"
