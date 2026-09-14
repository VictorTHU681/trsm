#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S) $(git log --oneline -1)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
rm -f trsm_test
gcc $FLAGS bench_trsm.c trsm.c -o trsm_test -lm -lkblas 2>&1 | grep -E "rror" | head -5 || true
[ -x ./trsm_test ] || { echo "BUILD FAILED"; exit 1; }
echo "--- 128x192 check"
TRSM_WGEMM=1 TRSM_WGEMM_CHECK=1 TRSM_OWNER=1 TRSM_MB=64 TRSM_NC=96 OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./trsm_test 128 192 1 2>&1 | grep -E "wchk|PASS|FAIL"
echo "--- 256x480 check"
TRSM_WGEMM=1 TRSM_WGEMM_CHECK=1 TRSM_OWNER=1 TRSM_MB=64 TRSM_NC=96 OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./trsm_test 256 480 1 2>&1 | grep -E "wchk|PASS|FAIL"
echo "--- 512x19968 check"
TRSM_WGEMM=1 TRSM_WGEMM_CHECK=1 OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./trsm_test 512 19968 1 2>&1 | grep -E "wchk|PASS|FAIL"
echo "=== DONE"
