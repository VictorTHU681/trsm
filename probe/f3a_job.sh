#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S) $(git log --oneline -1)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
rm -f trsm_test; gcc $FLAGS bench_trsm.c trsm.c -o trsm_test -lm -lkblas 2>&1 | grep -E "rror" | head -5
for v in "X=0" "GOMP_SPINCOUNT=infinite" "OMP_WAIT_POLICY=active" "GOMP_SPINCOUNT=100"; do for args in "17024 512 1" "512 19968 1"; do echo "[$v $args]"; env $v TRSM_PROFILE=2 OMP_NUM_THREADS=38 numactl -N 1 ./trsm_test $args 2>&1 | grep -E "唤醒|PASS" | tail -2 | cut -c1-120; done; done
echo "=== DONE $(date +%H:%M:%S)"
