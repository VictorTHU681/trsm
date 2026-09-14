#!/usr/bin/env bash
# sp87：XONE 后 SHAPEP/C0 重标定——旧惩罚表基于有装载相的任务成本，v63 相位结构已变，尾部 0.114 或有再平衡空间
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S) $(git log --oneline -1)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
rm -f trsm_sp87
gcc $FLAGS bench_trsm.c trsm.c -o trsm_sp87 -lm -lkblas 2>&1 | grep -E "rror" | head -3
[[ -x trsm_sp87 ]] || { echo "BUILD FAILED"; exit 1; }
run1(){ local lbl="$1"; shift; echo "--- $lbl"; env "$@" TRSM_PROFILE=2 OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./trsm_sp87 512 19968 5 2>&1 | grep -E " x |PASS|FAIL|每线程平均"; }
echo "=== 六线交错 x4"
for r in 1 2 3 4; do
  run1 "sp100_$r"
  run1 "sp0_$r"    TRSM_SHAPEP=0
  run1 "sp60_$r"   TRSM_SHAPEP=60
  run1 "sp140_$r"  TRSM_SHAPEP=140
  run1 "c0120_$r"  TRSM_OWNER_C0=120
  run1 "c0200_$r"  TRSM_OWNER_C0=200
done
rm -f trsm_sp87
echo "=== DONE $(date +%H:%M:%S)"
