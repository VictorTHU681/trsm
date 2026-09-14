#!/usr/bin/env bash
# cf62：折叠面板（TRSM_CFOLD，审计 #2）筛查
#   机制：B 基址 +16 → 每 768B 行跨 13 线、每 svld1 拆 2 次 L1 访问、每行 2 次部分行 NT 写（RMW）。
#   折叠后面板 1..207 全对齐（12 线/行、整行 NT 写），仅面板 0（1/208）保持错位。上界 = FB_XOFF +2.8%。
#   判读：load/store 双降且 task ≥+1% → 双节点正式认证；持平/负 → 关闭（对齐收益或被 MLP 墙吞掉）。
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S) $(git log --oneline -1)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
rm -f trsm_c62 fb_c62
gcc $FLAGS bench_trsm.c trsm.c -o trsm_c62 -lm -lkblas 2>&1 | grep -E "rror" | head -5
gcc $FLAGS probe/fastbench.c trsm.c -o fb_c62 -lm 2>&1 | grep -E "rror" | head -5
[[ -x trsm_c62 && -x fb_c62 ]] || { echo "BUILD FAILED"; exit 1; }
sha256sum trsm.c | cut -c1-40

echo "=== 正确性：官方 bench TC1 折叠开（全参考校验）"
TRSM_CFOLD=1 OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./trsm_c62 512 19968 3 2>&1 | grep -E " x |PASS|FAIL"
echo "=== 正确性：fastbench 折叠形状（抽样残差，-s 256）"
for sh in "512 9600" "448 7680" "960 19200"; do
  TRSM_CFOLD=1 OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./fb_c62 -r 2 -s 256 $sh 2>&1 | grep -vE "^  rep"
done
echo "=== 正确性：gen 回归（默认关 20 + 折叠开强制 owner 25）"
gcc -O3 -fopenmp -DTRSM_NO_SME_ASM -Dl_trsm=l_trsm_ref -c trsm.c -o probe/ref62.o 2>&1 | grep -E "error" || true
gcc $FLAGS probe/gen_test.c trsm.c probe/ref62.o -o probe/gen_test62 -lm 2>&1 | grep -E "error" || true
echo "[gen 默认 20]";                 timeout 600 numactl -N 1 ./probe/gen_test62 20 2>&1 | tail -1
echo "[gen CFOLD=1 强制owner 25]";    TRSM_CFOLD=1 TRSM_OWNER=1 timeout 600 numactl -N 1 ./probe/gen_test62 25 2>&1 | tail -1

echo "=== TC1 两线交错 x6（官方 bench + PROFILE=2）"
run1(){ local lbl="$1"; shift; echo "--- $lbl"; env "$@" TRSM_PROFILE=2 OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./trsm_c62 512 19968 5 2>&1 | grep -E " x |PASS|FAIL|每线程平均"; }
for r in 1 2 3 4 5 6; do
  run1 "base_$r"
  run1 "cf_$r" TRSM_CFOLD=1
done

echo "=== TC2/TC3 冒烟（CFOLD=1，owner 关闭故应不触发）"
TRSM_CFOLD=1 OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./trsm_c62 2432 17024 3 2>&1 | grep -E " x |PASS|FAIL"
TRSM_CFOLD=1 OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./trsm_c62 17024 512 3 2>&1 | grep -E " x |PASS|FAIL"
rm -f probe/ref62.o probe/gen_test62
echo "=== DONE $(date +%H:%M:%S)"
