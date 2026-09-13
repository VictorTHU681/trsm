#!/usr/bin/env bash
# SHAPEP 最大化：scale 90/100/110 + 深相位放大 HI 130/160（scale 100），同一二进制交错 ×8（官方 bench ×7 计时）
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S) $(git log --oneline -1)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
rm -f trsm_test
gcc $FLAGS bench_trsm.c trsm.c -o trsm_test -lm -lkblas 2>&1 | grep -E "rror" | head -5 || true
[ -x ./trsm_test ] || { echo "BUILD FAILED"; exit 1; }
run1() { local lbl="$1"; shift; echo "--- RUN $lbl"; env "$@" OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./trsm_test 512 19968 7 2>&1 | grep -E "512|PASS|FAIL"; }
for r in 1 2 3 4 5 6 7 8; do
  run1 "s100_$r"
  run1 "s90_$r"    TRSM_SHAPEP=90
  run1 "s110_$r"   TRSM_SHAPEP=110
  run1 "hi130_$r"  TRSM_SHAPEP_HI=130
  run1 "hi160_$r"  TRSM_SHAPEP_HI=160
done
echo "=== DONE $(date +%H:%M:%S)"
