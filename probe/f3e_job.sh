#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S) $(git log --oneline -1)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
rm -f trsm_test; gcc $FLAGS bench_trsm.c trsm.c -o trsm_test -lm -lkblas 2>&1 | grep -E "rror" | head -5
KB=$(ldd ./trsm_test | awk '/kblas/{print $3}'); strings $KB | grep -E "_NUM_THREADS|THREAD_TIMEOUT|_THREADS$|OMP_|GOMP" | sort -u | head -20; ldd $KB | grep -i "omp\|pthread"
for v in "X=0" "BLAS_THREAD_TIMEOUT=4" "BLAS_THREAD_TIMEOUT=30"; do for p in 0 1; do echo "[$v POOL=$p]"; env $v TRSM_POOL=$p TRSM_PROFILE=2 OMP_NUM_THREADS=38 timeout 300 numactl -N 1 ./trsm_test 17024 512 1 2>&1 | grep -E "唤醒|PASS|FAIL" | tail -2 | cut -c1-120; done; done
for i in 1 2 3; do for v in "X=0" "BLAS_THREAD_TIMEOUT=4"; do for args in "512 19968 1" "17024 512 1"; do printf "%-24s " "[$v]"; env $v OMP_NUM_THREADS=38 numactl -N 1 ./trsm_test $args 2>&1 | grep -E "PASS|FAIL"; done; done; done
echo "=== DONE $(date +%H:%M:%S)"
