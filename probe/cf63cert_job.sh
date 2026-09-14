#!/usr/bin/env bash
# cf63cert：CFOLD 正式认证（单节点份；提交两份到不同节点）——官方 bench 无 PROFILE，TC1 两线交错 x8 + TC2/TC3 x2
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S) $(git log --oneline -1)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
BIN="trsm_c63_$$"
gcc $FLAGS bench_trsm.c trsm.c -o $BIN -lm -lkblas 2>&1 | grep -E "rror" | head -5
[[ -x $BIN ]] || { echo "BUILD FAILED"; exit 1; }
run1(){ local lbl="$1"; shift; echo "--- $lbl"; env "$@" OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./$BIN ${ARGS} 2>&1 | grep -E " x |PASS|FAIL"; }
echo "=== TC1 两线 ABBA x8"
for r in 1 2 3 4; do
  ARGS="512 19968 5" run1 "tc1_base_a$r"
  ARGS="512 19968 5" run1 "tc1_cf_a$r"  TRSM_CFOLD=1
  ARGS="512 19968 5" run1 "tc1_cf_b$r"  TRSM_CFOLD=1
  ARGS="512 19968 5" run1 "tc1_base_b$r"
done
echo "=== TC2/TC3 两线 x2"
for r in 1 2; do
  for args in "2432 17024 3" "17024 512 3"; do
    ARGS="$args" run1 "tc_${args%% *}_base_$r"
    ARGS="$args" run1 "tc_${args%% *}_cf_$r" TRSM_CFOLD=1
  done
done
rm -f $BIN
echo "=== DONE $(date +%H:%M:%S)"
