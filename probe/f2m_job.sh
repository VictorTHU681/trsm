#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
rm -f trsm_test; gcc $FLAGS bench_trsm.c trsm.c -o trsm_test -lm -lkblas 2>&1 | grep -E "rror" | head -5
for c0 in 192 128 96 64 32; do echo "[C0=$c0]"; TRSM_OWNER_C0=$c0 TRSM_PROFILE=2 OMP_NUM_THREADS=38 numactl -N 1 ./trsm_test 512 19968 1 2>&1 | grep -E "尾部" | sed 's/.*task 合计/task 合计/'; done
echo "--- bench (no profile), 3 rounds each"
for i in 1 2 3; do for c0 in 192 128 96 64 32; do printf "[C0=%s] " $c0; TRSM_OWNER_C0=$c0 OMP_NUM_THREADS=38 numactl -N 1 ./trsm_test 512 19968 1 2>&1 | grep -E "PASS|FAIL"; done; done
echo "=== DONE $(date +%H:%M:%S)"
