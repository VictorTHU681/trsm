#!/usr/bin/env bash
# TC2/TC3 每线程分相诊断（TRSM_PROFILE=4）：定位墙钟 − 分相和的差距（尾部/派发/flag 等待的线程分布）
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S) $(git log --oneline -1)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
rm -f trsm_test
gcc $FLAGS bench_trsm.c trsm.c -o trsm_test -lm -lkblas 2>&1 | grep -E "rror" | head -5 || true
[ -x ./trsm_test ] || { echo "BUILD FAILED"; exit 1; }
echo "--- ROUND tc2"
TRSM_PROFILE=4 OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./trsm_test 2432 17024 3 2>&1
echo "--- ROUND tc3"
TRSM_PROFILE=4 OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./trsm_test 17024 512 3 2>&1
echo "=== DONE $(date +%H:%M:%S)"
