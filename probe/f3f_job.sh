#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S) $(git log --oneline -1)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
rm -f trsm_test; gcc $FLAGS bench_trsm.c trsm.c -o trsm_test -lm -lkblas 2>&1 | grep -E "rror" | head -5
CFGS=("X=0" "TRSM_DNT=0" "TRSM_LOADPF=16" "TRSM_LOADPF=8" "TRSM_DNT=0 TRSM_LOADPF=16" "TRSM_OWNER_C0=192")
for i in 1 2 3 4; do for v in "${CFGS[@]}"; do printf "%-30s " "[$v]"; env $v OMP_NUM_THREADS=38 numactl -N 1 ./trsm_test 512 19968 1 2>&1 | grep -E "PASS|FAIL"; done; done
for v in "X=0" "TRSM_DNT=0" "TRSM_LOADPF=16"; do echo "[$v]"; env $v TRSM_PROFILE=2 OMP_NUM_THREADS=38 numactl -N 1 ./trsm_test 512 19968 1 2>&1 | grep -E "每线程平均" | tail -1 | cut -c1-330; done
echo "=== DONE $(date +%H:%M:%S)"
