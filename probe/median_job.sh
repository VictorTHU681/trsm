#!/usr/bin/env bash
# 官方 bench 多轮中位数（判题 run.sh 口径：--preferred）：TC1 ×4、TC2 ×8、TC3 ×4（交错）
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S) $(git log --oneline -1)"
. probe/env.sh
uptime
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
rm -f trsm_test; gcc $FLAGS bench_trsm.c trsm.c -o trsm_test -lm -lkblas 2>&1 | grep -E "rror" | head -5
for i in 1 2 3 4; do
  for args in "512 19968 1" "2432 17024 1" "2432 17024 1" "17024 512 1"; do
    OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./trsm_test $args 2>&1 | grep -E "PASS|FAIL"
  done
done
echo "=== DONE $(date +%H:%M:%S)"
