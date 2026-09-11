#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
rm -f trsm_test; gcc $FLAGS bench_trsm.c trsm.c -o trsm_test -lm -lkblas 2>&1 | grep -E "rror" | head -5
for nt in 1 2 4 8 16 24 32 38; do echo "[nth=$nt case2]"; TRSM_PROFILE=2 OMP_NUM_THREADS=$nt numactl -N 1 ./trsm_test 2432 17024 1 2>&1 | grep -E "每线程|PASS|FAIL|mode" | tail -3; done
for nt in 1 4 16 38; do echo "[nth=$nt case3]"; TRSM_PROFILE=2 OMP_NUM_THREADS=$nt numactl -N 1 ./trsm_test 17024 512 1 2>&1 | grep -E "每线程|PASS|FAIL|mode" | tail -3; done
echo "=== DONE $(date +%H:%M:%S)"
