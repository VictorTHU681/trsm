#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S) $(git log --oneline -1)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
rm -f trsm_test; gcc $FLAGS bench_trsm.c trsm.c -o trsm_test -lm -lkblas 2>&1 | grep -E "rror" | head -5
CFGS=("TRSM_TAILW_SEG=1" "TRSM_TAILW_SEG=2" "TRSM_TAILW_SEG=3" "TRSM_TAILW_ROWS=2 TRSM_TAILW_SEG=2")
for i in 1 2 3; do for v in "${CFGS[@]}"; do printf "%-36s " "[$v]"; env $v OMP_NUM_THREADS=38 numactl -N 1 ./trsm_test 2432 17024 1 2>&1 | grep -E "PASS|FAIL"; done; done
for v in "TRSM_TAILW_SEG=1" "TRSM_TAILW_SEG=2"; do echo "[$v]"; env $v TRSM_PROFILE=2 OMP_NUM_THREADS=38 numactl -N 1 ./trsm_test 2432 17024 1 2>&1 | grep -E "每线程平均" | tail -1 | cut -c1-330; done
echo "=== DONE $(date +%H:%M:%S)"
