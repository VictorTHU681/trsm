#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S) $(git log --oneline -1)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
rm -f trsm_test trsm_test_nt; gcc $FLAGS bench_trsm.c trsm.c -o trsm_test -lm -lkblas 2>&1 | grep -E "rror" | head -5
gcc $FLAGS -DTRSM_A4NT bench_trsm.c trsm.c -o trsm_test_nt -lm -lkblas 2>&1 | grep -E "rror" | head -5
for i in 1 2 3 4; do for b in trsm_test trsm_test_nt; do for args in "2432 17024 1" "17024 512 1"; do printf "%-14s " "[$b]"; OMP_NUM_THREADS=38 numactl -N 1 ./$b $args 2>&1 | grep -E "PASS|FAIL"; done; done; done
for b in trsm_test trsm_test_nt; do echo "[$b]"; TRSM_PROFILE=2 OMP_NUM_THREADS=38 numactl -N 1 ./$b 2432 17024 1 2>&1 | grep -E "每线程平均" | tail -1 | cut -c1-330; done
echo "=== DONE $(date +%H:%M:%S)"
