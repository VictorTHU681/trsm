#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
rm -f trsm_test; gcc $FLAGS bench_trsm.c trsm.c -o trsm_test -lm -lkblas 2>&1 | grep -E "rror" | head -5
for i in 1 2; do TRSM_PROFILE=2 OMP_NUM_THREADS=38 numactl -N 1 ./trsm_test 512 19968 1 2>&1 | grep -E "每线程|PASS|prof\]" | tail -4; done
echo "--- nth scan case1"
for nt in 1 4 19 38; do echo "[nth=$nt]"; TRSM_PROFILE=2 OMP_NUM_THREADS=$nt numactl -N 1 ./trsm_test 512 19968 1 2>&1 | grep -E "每线程|PASS" | tail -2; done
echo "=== DONE $(date +%H:%M:%S)"
