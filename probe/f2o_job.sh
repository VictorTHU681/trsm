#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
rm -f trsm_test; gcc $FLAGS bench_trsm.c trsm.c -o trsm_test -lm -lkblas 2>&1 | grep -E "rror" | head -5
CFGS=("TRSM_PACK_LA=2" "TRSM_PACK_LA=3" "TRSM_PACK_LA=5" "TRSM_TAIL_SEG=8" "TRSM_TAIL_SEG=12 TRSM_MAXSEG=12" "TRSM_TAIL_ROWS=3" "TRSM_TAIL_ROWS=4 TRSM_TAIL_SEG=8 TRSM_MAXSEG=12" "TRSM_SPLITF=16" "TRSM_SPLITF=24 TRSM_MAXSEG=12")
for i in 1 2 3; do for v in "${CFGS[@]}"; do printf "%-48s " "[$v]"; env $v OMP_NUM_THREADS=38 numactl -N 1 ./trsm_test 17024 512 1 2>&1 | grep -E "PASS|FAIL"; done; done
echo "--- profile"
for v in "TRSM_PACK_LA=2" "TRSM_PACK_LA=5" "TRSM_TAIL_ROWS=4 TRSM_TAIL_SEG=8 TRSM_MAXSEG=12"; do echo "[$v]"; env $v TRSM_PROFILE=2 OMP_NUM_THREADS=38 numactl -N 1 ./trsm_test 17024 512 1 2>&1 | grep -E "每线程平均" | tail -1 | cut -c1-330; done
echo "=== DONE $(date +%H:%M:%S)"
