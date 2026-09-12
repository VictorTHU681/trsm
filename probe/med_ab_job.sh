#!/usr/bin/env bash
# v48 NEXTA 官方 bench 配对 A/B：同节点、同轮交替 TRSM_NEXTA=0（v47 行为）/ 默认开
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S) $(git log --oneline -1)"
. probe/env.sh
uptime
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
rm -f trsm_test; gcc $FLAGS bench_trsm.c trsm.c -o trsm_test -lm -lkblas 2>&1 | grep -E "rror" | head -5
run() { OMP_NUM_THREADS=38 numactl -N 1 ./trsm_test $2 $3 1 2>&1 | grep -E "PASS|FAIL" | sed "s/^/[$1] /"; }
for i in 1 2 3 4 5 6; do
  TRSM_NEXTA=0 run off 512 19968
  run on 512 19968
done
for i in 1 2 3 4; do
  TRSM_NEXTA=0 run off 2432 17024
  run on 2432 17024
  TRSM_NEXTA=0 run off 17024 512
  run on 17024 512
done
echo "=== DONE $(date +%H:%M:%S)"
