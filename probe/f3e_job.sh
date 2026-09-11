#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S) $(git log --oneline -1)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
rm -f trsm_test; gcc $FLAGS bench_trsm.c trsm.c -o trsm_test -lm -lkblas 2>&1 | grep -E "rror" | head -5
for p in 1 0; do echo "[POOL=$p TC3]"; TRSM_CENSUS=1 TRSM_POOL=$p TRSM_PROFILE=3 OMP_NUM_THREADS=38 timeout 300 numactl -N 1 ./trsm_test 17024 512 2 2>&1 | grep -E "census|唤醒|PASS|FAIL" | cut -c1-900; done
echo "[POOL=1 TC1]"; TRSM_CENSUS=1 TRSM_POOL=1 TRSM_PROFILE=3 OMP_NUM_THREADS=38 timeout 300 numactl -N 1 ./trsm_test 512 19968 2 2>&1 | grep -E "census|唤醒|PASS|FAIL" | cut -c1-900
echo "=== DONE $(date +%H:%M:%S)"
