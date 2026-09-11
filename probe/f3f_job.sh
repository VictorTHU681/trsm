#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S) $(git log --oneline -1)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
rm -f trsm_test; gcc $FLAGS bench_trsm.c trsm.c -o trsm_test -lm -lkblas 2>&1 | grep -E "rror" | head -5
for i in 1 2; do TRSM_PROFILE=3 OMP_NUM_THREADS=38 numactl -N 1 ./trsm_test 512 19968 1 2>&1 | grep -E "prof\] m=|每线程平均|尾部|每线程收尾|按行块|PASS" | tail -6 | cut -c1-700; done
echo "=== DONE $(date +%H:%M:%S)"
