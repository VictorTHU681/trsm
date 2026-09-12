#!/usr/bin/env bash
# M8 正确性：special5 owner 形状（m%64==0, m≤1024, n%96==0, n≥7296），全矩阵比对
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(git log --oneline -1)"
. probe/env.sh
FLAGS="-O2 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
rm -f probe/nc256_dbg
gcc $FLAGS probe/nc256_dbg.c trsm.c -o probe/nc256_dbg -lm 2>&1 | grep -E "rror" | head -5
[[ -x probe/nc256_dbg ]] || { echo COMPILE_FAIL; exit 1; }
for env in "TRSM_M8=1" "TRSM_M8=0"; do
  for mn in "512 19968" "512 9600" "1024 9600" "960 9600" "256 7680"; do
    echo "--- [$env] m n = $mn"
    env $env OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./probe/nc256_dbg $mn 2>&1 | head -4
  done
done
echo "=== DONE"
