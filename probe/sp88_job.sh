#!/usr/bin/env bash
# sp88：sp0 vs sp100 正式裁决（XONE 后 SHAPEP 存废）——官方 ABBA x8
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S) $(git log --oneline -1)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
BIN=trsm_sp88_$$
gcc $FLAGS bench_trsm.c trsm.c -o $BIN -lm -lkblas 2>&1 | grep -E "rror" | head -3
[[ -x $BIN ]] || { echo "BUILD FAILED"; exit 1; }
run1(){ local lbl="$1"; shift; echo "--- $lbl"; env "$@" OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./$BIN 512 19968 5 2>&1 | grep -E " x |PASS|FAIL"; }
for r in 1 2 3 4; do
  run1 "sp100_a$r"
  run1 "sp0_a$r" TRSM_SHAPEP=0
  run1 "sp0_b$r" TRSM_SHAPEP=0
  run1 "sp100_b$r"
done
rm -f $BIN
echo "=== DONE $(date +%H:%M:%S)"
