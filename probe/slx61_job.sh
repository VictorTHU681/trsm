#!/usr/bin/env bash
# slx61：SLX 块交错粒度扫描（逐行版已证伪：ROB 稀释 → 净负 3.6%；块版预期 load 突发恢复 MLP）
#   判读：若 slxbN 的 (store+load) < base 的 0.424 且 task 净降 ≥1.5% → 进正式认证；全负 → SLX 方向关闭
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S) $(git log --oneline -1)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
rm -f trsm_s61
gcc $FLAGS bench_trsm.c trsm.c -o trsm_s61 -lm -lkblas 2>&1 | grep -E "rror" | head -5
[[ -x trsm_s61 ]] || { echo "BUILD FAILED"; exit 1; }
sha256sum trsm.c | cut -c1-40

echo "=== gen 冒烟（块重构后 off/on 两路）"
gcc -O3 -fopenmp -DTRSM_NO_SME_ASM -Dl_trsm=l_trsm_ref -c trsm.c -o probe/ref61.o 2>&1 | grep -E "error" || true
gcc $FLAGS probe/gen_test.c trsm.c probe/ref61.o -o probe/gen_test61 -lm 2>&1 | grep -E "error" || true
echo "[gen SLX=0 20]";           timeout 600 numactl -N 1 ./probe/gen_test61 20 2>&1 | tail -1
echo "[gen SLX=1 强制owner 25]"; TRSM_SLX=1 TRSM_OWNER=1 timeout 600 numactl -N 1 ./probe/gen_test61 25 2>&1 | tail -1

echo "=== TC1 五线交错 x4（base / slxb 4/8/16/32）"
run1(){ local lbl="$1"; shift; echo "--- $lbl"; env "$@" TRSM_PROFILE=2 OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./trsm_s61 512 19968 5 2>&1 | grep -E " x |PASS|FAIL|每线程平均"; }
for r in 1 2 3 4; do
  run1 "base_$r"
  run1 "b4_$r"  TRSM_SLX=1 TRSM_SLXB=4
  run1 "b8_$r"  TRSM_SLX=1 TRSM_SLXB=8
  run1 "b16_$r" TRSM_SLX=1 TRSM_SLXB=16
  run1 "b32_$r" TRSM_SLX=1 TRSM_SLXB=32
done
rm -f probe/ref61.o probe/gen_test61
echo "=== DONE $(date +%H:%M:%S)"
