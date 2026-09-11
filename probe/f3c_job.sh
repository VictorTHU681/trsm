#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S) $(git log --oneline -1)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
rm -f trsm_test trsm_test_master; gcc $FLAGS bench_trsm.c trsm.c -o trsm_test -lm -lkblas 2>&1 | grep -E "rror" | head -5
git show master:trsm.c > /tmp/trsm_master.c && gcc $FLAGS bench_trsm.c /tmp/trsm_master.c -o trsm_test_master -lm -lkblas 2>&1 | grep -E "rror" | head -5
for i in 1 2 3 4 5 6; do for b in trsm_test_master trsm_test; do for args in "512 19968 1" "2432 17024 1" "17024 512 1"; do printf "%-18s " "[$b]"; OMP_NUM_THREADS=38 numactl -N 1 ./$b $args 2>&1 | grep -E "PASS|FAIL"; done; done; done
echo "=== DONE $(date +%H:%M:%S)"
