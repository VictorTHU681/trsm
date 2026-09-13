#!/usr/bin/env bash
# SHAPEP A/B 筛选：同一二进制 TRSM_SHAPEP=0/100 交错 ×6 + 档位 50/150 ×2 + 机制验证（PROFILE=4 各一次）
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S) $(git log --oneline -1)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
rm -f trsm_test
gcc $FLAGS bench_trsm.c trsm.c -o trsm_test -lm -lkblas 2>&1 | grep -E "rror" | head -5 || true
[ -x ./trsm_test ] || { echo "BUILD FAILED"; exit 1; }
run1() { local lbl="$1"; shift; echo "--- RUN $lbl"; env "$@" OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./trsm_test 512 19968 7 2>&1 | grep -E "512|PASS|FAIL"; }
for r in 1 2 3 4 5 6; do
  run1 "off_$r"  TRSM_SHAPEP=0
  run1 "on_$r"   TRSM_SHAPEP=100
done
run1 s50_1  TRSM_SHAPEP=50
run1 s150_1 TRSM_SHAPEP=150
run1 s50_2  TRSM_SHAPEP=50
run1 s150_2 TRSM_SHAPEP=150
echo "--- PROFILE off"
env TRSM_SHAPEP=0   TRSM_PROFILE=4 OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./trsm_test 512 19968 5 2>&1 | tail -90
echo "--- PROFILE on"
env TRSM_SHAPEP=100 TRSM_PROFILE=4 OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./trsm_test 512 19968 5 2>&1 | tail -90
echo "=== DONE $(date +%H:%M:%S)"
