#!/usr/bin/env bash
# prof63：v63 终态分相地图（三用例 x4，PROFILE=2）
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S) $(git log --oneline -1)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
rm -f trsm_p63
gcc $FLAGS bench_trsm.c trsm.c -o trsm_p63 -lm -lkblas 2>&1 | grep -E "rror" | head -3
[[ -x trsm_p63 ]] || { echo "BUILD FAILED"; exit 1; }
for args in "512 19968 5" "2432 17024 5" "17024 512 5"; do
  for r in 1 2 3 4; do
    echo "--- tc_${args%% *}_$r"
    env TRSM_PROFILE=2 OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./trsm_p63 $args 2>&1 | grep -E " x |PASS|FAIL|每线程平均"
  done
done
rm -f trsm_p63
echo "=== DONE $(date +%H:%M:%S)"
