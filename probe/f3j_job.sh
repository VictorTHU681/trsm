#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S) $(git log --oneline -1)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
rm -f trsm_test; gcc $FLAGS bench_trsm.c trsm.c -o trsm_test -lm -lkblas 2>&1 | grep -E "rror" | head -5
for i in 1 2 3 4; do for v in 1 0; do for args in "512 19968 1" "2432 17024 1"; do printf "[LSPLIT=%s] " $v; TRSM_LSPLIT=$v OMP_NUM_THREADS=38 timeout 300 numactl -N 1 ./trsm_test $args 2>&1 | grep -E "PASS|FAIL"; done; done; done
for v in 1 0; do echo "[LSPLIT=$v case2]"; TRSM_LSPLIT=$v TRSM_PROFILE=2 OMP_NUM_THREADS=38 timeout 300 numactl -N 1 ./trsm_test 2432 17024 1 2>&1 | grep -E "每线程平均|prof\] m=" | tail -2 | cut -c1-330; done
echo "=== DONE $(date +%H:%M:%S)"
