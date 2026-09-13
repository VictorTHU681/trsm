#!/usr/bin/env bash
# 宽面板 c256 现码复测：fresh11 二进制 TC3 nc=256(自动) vs 强制 nc=128，官方 bench，交错 ×4 + 分相
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S) $(git log --oneline -1)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
rm -f trsm_test
gcc $FLAGS bench_trsm.c trsm.c -o trsm_test -lm -lkblas 2>&1 | grep -E "rror" | head -5 || true
[ -x ./trsm_test ] || { echo "BUILD FAILED"; exit 1; }
run1() { local lbl="$1"; shift; echo "--- RUN $lbl"; env "$@" OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./trsm_test 17024 512 5 2>&1 | grep -E "17024|PASS|FAIL"; }
for r in 1 2 3 4; do
  run1 "nc128_$r" TRSM_NC=128
  run1 "nc256_$r"
done
echo "--- 分相 nc128"; env TRSM_NC=128 TRSM_PROFILE=2 OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./trsm_test 17024 512 3 2>&1 | grep -E "每线程平均|prof\]" | tail -3
echo "--- 分相 nc256"; env TRSM_PROFILE=2 OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./trsm_test 17024 512 3 2>&1 | grep -E "每线程平均|prof\]" | tail -3
echo "=== DONE $(date +%H:%M:%S)"
