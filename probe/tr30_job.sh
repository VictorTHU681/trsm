#!/usr/bin/env bash
# tr30：test_runs 10 vs 30 官方均值对照（冷轮稀释账）+ PROFILE=4 输出样本（SHAPEP 残尾诊断格式勘察）
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S) $(git log --oneline -1)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
rm -f trsm_tr30
gcc $FLAGS bench_trsm.c trsm.c -o trsm_tr30 -lm -lkblas 2>&1 | grep -E "rror" | head -3
[[ -x trsm_tr30 ]] || { echo "BUILD FAILED"; exit 1; }
run1(){ local lbl="$1" args="$2"; echo "--- $lbl"; OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./trsm_tr30 $args 2>&1 | grep -E " x |PASS|FAIL"; }
for r in 1 2 3; do
  run1 "tc1_r10_$r" "512 19968 10"
  run1 "tc1_r30_$r" "512 19968 30"
  run1 "tc2_r10_$r" "2432 17024 10"
  run1 "tc2_r30_$r" "2432 17024 30"
  run1 "tc3_r10_$r" "17024 512 10"
  run1 "tc3_r30_$r" "17024 512 30"
done
echo "=== PROFILE=4 样本（TC1，看输出格式）"
TRSM_PROFILE=4 OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./trsm_tr30 512 19968 2 2>&1 | tail -40
rm -f trsm_tr30
echo "=== DONE $(date +%H:%M:%S)"
