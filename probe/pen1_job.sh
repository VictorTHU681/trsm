#!/usr/bin/env bash
# pen1：SHAPEP 惩罚表扫参（prof4 残尾诊断 → ia=2 提额候选）——TC1 五线交错 x4
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S) $(git log --oneline -1)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
rm -f trsm_pen1
gcc $FLAGS bench_trsm.c trsm.c -o trsm_pen1 -lm -lkblas 2>&1 | grep -E "rror" | head -3
[[ -x trsm_pen1 ]] || { echo "BUILD FAILED"; exit 1; }
run1(){ local lbl="$1"; shift; echo "--- $lbl"; env "$@" TRSM_PROFILE=2 OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./trsm_pen1 512 19968 4 2>&1 | grep -E " x |PASS|FAIL|每线程平均"; }
for r in 1 2 3 4; do
  run1 "base_$r"
  run1 "p2a_$r"  TRSM_PENV="0,56,70,60,70,149,112,78"
  run1 "p2b_$r"  TRSM_PENV="0,56,100,60,70,149,112,78"
  run1 "p2c_$r"  TRSM_PENV="0,70,90,75,85,149,112,85"
  run1 "p2d_$r"  TRSM_PENV="0,56,120,60,70,120,100,78"
done
rm -f trsm_pen1
echo "=== DONE $(date +%H:%M:%S)"
