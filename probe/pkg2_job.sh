#!/usr/bin/env bash
# 双包自包含验收：解压 → bash run.sh（不注入任何文件），检查三行 PASS；诊断包另查诊断段
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S)"
for z in ~/S2/trsm_submit_v54_*.zip ~/S2/trsm_submit_diag_*.zip; do
    [[ -f "$z" ]] || { echo "MISSING $z"; continue; }
    echo "===== 包：$(basename "$z") $(date +%H:%M:%S)"
    W=/tmp/pkgv_$$_$RANDOM
    rm -rf "$W"; mkdir -p "$W"
    unzip -q "$z" -d "$W" || { echo "UNZIP FAIL"; continue; }
    ( cd "$W/trsm" && timeout -k 10 1200 bash run.sh > out.txt 2> err.txt; echo "exit=$?" )
    grep -cE "PASS" "$W/trsm/out.txt" | xargs echo "PASS 行数:"
    grep -E "FAIL|rror" "$W/trsm/out.txt" "$W/trsm/err.txt" | grep -v "最大误差" | head -5
    grep -E "GFLOPS|PASS" "$W/trsm/out.txt" | head -8
    d=$(awk '/\[trsm-diag\]/{f=1} f' "$W/trsm/out.txt" | wc -l)
    echo "诊断段行数: $d"
    rm -rf "$W"
done
echo "=== DONE $(date +%H:%M:%S)"
