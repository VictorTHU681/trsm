#!/usr/bin/env bash
# REHUGE 开/关配对（题面原样命令，无 --preferred——REHUGE 只在此路径生效；两臂 X 迁移均开）
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S) $(git log --oneline -1)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
rm -f trsm_test; gcc $FLAGS bench_trsm.c trsm.c -o trsm_test -lm -lkblas 2>&1 | grep -E "rror" | head -5
for i in 1 2 3; do
  for MN in "512 19968" "2432 17024" "17024 512"; do
    read -r M N <<<"$MN"
    TRSM_REHUGE=0 OMP_NUM_THREADS=38 numactl -N 1 ./trsm_test $M $N 1 2>&1 | grep -E "PASS|FAIL" | sed "s/^/[off] /"
    OMP_NUM_THREADS=38 numactl -N 1 ./trsm_test $M $N 1 2>&1 | grep -E "PASS|FAIL" | sed "s/^/[on ] /"
  done
done
echo "--- THP 覆盖对比（TC1，off / on）"
TRSM_SMAPS=1 TRSM_REHUGE=0 OMP_NUM_THREADS=38 numactl -N 1 ./trsm_test 512 19968 1 2>&1 | grep -E "rehuge|smaps" | head -5
TRSM_SMAPS=1 OMP_NUM_THREADS=38 numactl -N 1 ./trsm_test 512 19968 1 2>&1 | grep -E "rehuge|smaps" | head -5
echo "=== DONE $(date +%H:%M:%S)"
