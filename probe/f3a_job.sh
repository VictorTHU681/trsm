#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S) $(git log --oneline -1)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
rm -f trsm_test; gcc $FLAGS bench_trsm.c trsm.c -o trsm_test -lm -lkblas 2>&1 | grep -E "rror" | head -5
for args in "512 19968 1" "2432 17024 1" "17024 512 1"; do echo "[$args]"; TRSM_PROFILE=2 OMP_NUM_THREADS=38 numactl -N 1 ./trsm_test $args 2>&1 | grep -E "唤醒|setup|PASS" | cut -c1-200; done
echo "[TC1 OMP_WAIT_POLICY=active]"; OMP_WAIT_POLICY=active TRSM_PROFILE=2 OMP_NUM_THREADS=38 numactl -N 1 ./trsm_test 512 19968 1 2>&1 | grep -E "唤醒|setup|PASS" | cut -c1-200
for i in 1 2 3; do for v in "OMP_WAIT_POLICY=passive" "OMP_WAIT_POLICY=active" "GOMP_SPINCOUNT=infinite"; do printf "%-26s " "[$v]"; env $v OMP_NUM_THREADS=38 numactl -N 1 ./trsm_test 512 19968 1 2>&1 | grep -E "PASS|FAIL"; done; done
echo "=== DONE $(date +%H:%M:%S)"
