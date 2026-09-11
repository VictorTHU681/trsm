#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S) $(git log --oneline -1)"
. probe/env.sh
cat /sys/kernel/mm/transparent_hugepage/enabled /sys/kernel/mm/transparent_hugepage/defrag; uname -r
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
rm -f trsm_test; gcc $FLAGS bench_trsm.c trsm.c -o trsm_test -lm -lkblas 2>&1 | grep -E "rror" | head -5
for args in "2432 17024 1" "17024 512 1" "512 19968 1"; do echo "[$args]"; TRSM_SMAPS=1 OMP_NUM_THREADS=38 numactl -N 1 ./trsm_test $args 2>&1 | grep -E "smaps|PASS|FAIL" | tail -8; done
echo "=== DONE $(date +%H:%M:%S)"
