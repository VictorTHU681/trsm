#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
rm -f trsm_test; gcc $FLAGS bench_trsm.c trsm.c -o trsm_test -lm -lkblas 2>&1 | grep -E "rror" | head -5
for i in 1 2 3; do for v in "TRSM_MB=192" "TRSM_MB=256"; do printf "%-14s " "[$v]"; env $v OMP_NUM_THREADS=38 numactl -N 1 ./trsm_test 17024 512 1 2>&1 | grep -E "PASS|FAIL"; done; done
for v in "TRSM_MB=192" "TRSM_MB=256"; do echo "[$v]"; env $v TRSM_PROFILE=2 OMP_NUM_THREADS=38 numactl -N 1 ./trsm_test 17024 512 1 2>&1 | grep -E "每线程平均|prof\] m=" | tail -2 | cut -c1-330; done
echo "=== DONE $(date +%H:%M:%S)"
