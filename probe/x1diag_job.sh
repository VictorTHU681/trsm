#!/usr/bin/env bash
# x1diag：XONE 在官方 TC1 失败的变量隔离（CFOLD? STEAL? 形状类?）
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S) $(git log --oneline -1)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
rm -f trsm_x1d fb_x1d
gcc $FLAGS bench_trsm.c trsm.c -o trsm_x1d -lm -lkblas 2>&1 | grep -E "rror" | head -3
gcc $FLAGS probe/fastbench.c trsm.c -o fb_x1d -lm 2>&1 | grep -E "rror" | head -3
[[ -x trsm_x1d && -x fb_x1d ]] || { echo "BUILD FAILED"; exit 1; }
B="TRSM_OWNER_CURSOR=0 TRSM_XONE=1"
run1(){ local lbl="$1"; shift; echo "--- $lbl"; env "$@" OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./trsm_x1d 512 19968 2 2>&1 | grep -E " x |PASS|FAIL"; }
echo "=== 官方 TC1 变量隔离"
run1 "xone_full"       TRSM_OWNER_CURSOR=0 TRSM_XONE=1
run1 "xone_nocfold"    TRSM_OWNER_CURSOR=0 TRSM_XONE=1 TRSM_CFOLD=0
run1 "xone_nosteal"    TRSM_OWNER_CURSOR=0 TRSM_XONE=1 TRSM_STEAL=0
run1 "xone_neither"    TRSM_OWNER_CURSOR=0 TRSM_XONE=1 TRSM_CFOLD=0 TRSM_STEAL=0
run1 "base_ctrl"       TRSM_OWNER_CURSOR=0
echo "=== fastbench 形状类对照（-s 256 抽样残差）"
for sh in "512 9600" "512 9500" "256 7680" "512 19968"; do
  echo "--- fb $sh"; env TRSM_OWNER_CURSOR=0 TRSM_XONE=1 OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./fb_x1d -r 1 -s 256 $sh 2>&1 | grep -vE "^  rep"
done
rm -f trsm_x1d fb_x1d
echo "=== DONE $(date +%H:%M:%S)"
