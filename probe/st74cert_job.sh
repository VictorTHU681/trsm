#!/usr/bin/env bash
# st74cert：STEAL v2 干净节点认证（单节点份，提交两份）——official bench TC1 ABBA x8 + TC2/TC3 冒烟
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S) $(git log --oneline -1)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
BIN="trsm_st74_$$"
gcc $FLAGS bench_trsm.c trsm.c -o $BIN -lm -lkblas 2>&1 | grep -E "rror" | head -3
[[ -x $BIN ]] || { echo "BUILD FAILED"; exit 1; }
run1(){ local lbl="$1"; shift; echo "--- $lbl"; env "$@" OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./$BIN ${ARGS} 2>&1 | grep -E " x |PASS|FAIL"; }
echo "=== TC1 ABBA x8"
for r in 1 2 3 4; do
  ARGS="512 19968 5" run1 "tc1_off_a$r"
  ARGS="512 19968 5" run1 "tc1_st_a$r" TRSM_STEAL=1
  ARGS="512 19968 5" run1 "tc1_st_b$r" TRSM_STEAL=1
  ARGS="512 19968 5" run1 "tc1_off_b$r"
done
echo "=== TC2/TC3 冒烟（owner 不触发，应逐位一致）"
for args in "2432 17024 3" "17024 512 3"; do
  ARGS="$args" run1 "tc_${args%% *}_off" 
  ARGS="$args" run1 "tc_${args%% *}_st" TRSM_STEAL=1
done
rm -f $BIN
echo "=== DONE $(date +%H:%M:%S)"
