#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
rm -f trsm_test; gcc $FLAGS bench_trsm.c trsm.c -o trsm_test -lm -lkblas 2>&1 | grep -E "rror" | head -5
for v in "TRSM_OWNER_C0=192 TRSM_OWNER_C2=0" "TRSM_OWNER_C0=371 TRSM_OWNER_C2=28" "TRSM_OWNER_C0=300 TRSM_OWNER_C2=20" "TRSM_OWNER_C0=371 TRSM_OWNER_C2=40"; do echo "[$v]"; env $v TRSM_PROFILE=3 OMP_NUM_THREADS=38 numactl -N 1 ./trsm_test 512 19968 1 2>&1 | grep -E "尾部|每线程收尾" | tail -2 | sed 's/.*task 合计/task 合计/'; done
echo "--- bench 5 rounds interleaved"
for i in 1 2 3 4 5; do for v in "TRSM_OWNER_C0=192 TRSM_OWNER_C2=0" "TRSM_OWNER_C0=371 TRSM_OWNER_C2=28" "TRSM_OWNER_C0=300 TRSM_OWNER_C2=20"; do printf "[%s] " "$v"; env $v OMP_NUM_THREADS=38 numactl -N 1 ./trsm_test 512 19968 1 2>&1 | grep -E "PASS|FAIL"; done; done
echo "=== DONE $(date +%H:%M:%S)"
