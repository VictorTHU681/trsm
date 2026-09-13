#!/usr/bin/env bash
# SHAPEP 细化筛选：LS 局部相位搜索 + C0 重标定（SHAPEP=100 底座，同一二进制交错 4 轮）
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S) $(git log --oneline -1)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
rm -f trsm_test
gcc $FLAGS bench_trsm.c trsm.c -o trsm_test -lm -lkblas 2>&1 | grep -E "rror" | head -5 || true
[ -x ./trsm_test ] || { echo "BUILD FAILED"; exit 1; }
run1() { local lbl="$1"; shift; echo "--- RUN $lbl"; env "$@" OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./trsm_test 512 19968 7 2>&1 | grep -E "512|PASS|FAIL"; }
for r in 1 2 3 4; do
  run1 "base_$r"
  run1 "ls2_$r"     TRSM_SHAPEP_LS=2
  run1 "ls4_$r"     TRSM_SHAPEP_LS=4
  run1 "c100_$r"    TRSM_OWNER_C0=100
  run1 "c130_$r"    TRSM_OWNER_C0=130
  run1 "c100ls2_$r" TRSM_OWNER_C0=100 TRSM_SHAPEP_LS=2
done
echo "=== DONE $(date +%H:%M:%S)"
