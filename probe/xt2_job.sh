#!/usr/bin/env bash
# xt2：XONE-TC2 筛查——装载相（0.83ms/线程）融入 chunk-0 旁流（duplex 路径验证机制，赢了再移植 hoist）
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S) $(git log --oneline -1)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
rm -f trsm_xt2
gcc $FLAGS bench_trsm.c trsm.c -o trsm_xt2 -lm -lkblas 2>&1 | grep -E "rror" | head -5
[[ -x trsm_xt2 ]] || { echo "BUILD FAILED"; exit 1; }
echo "=== gen 回归（默认全开含 TC1-XONE）"
gcc -O3 -fopenmp -DTRSM_NO_SME_ASM -Dl_trsm=l_trsm_ref -c trsm.c -o probe/refxt.o 2>&1 | grep -E "error" || true
gcc $FLAGS probe/gen_test.c trsm.c probe/refxt.o -o probe/gen_xt -lm 2>&1 | grep -E "error" || true
echo "[gen 默认 30]"; timeout 700 numactl -N 1 ./probe/gen_xt 30 2>&1 | tail -1
run2(){ local lbl="$1"; shift; echo "--- $lbl"; env "$@" TRSM_PROFILE=2 OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./trsm_xt2 2432 17024 5 2>&1 | grep -E " x |PASS|FAIL|每线程平均"; }
echo "=== TC2 三线交错 x5（hoist 生产 / duplex 基线 / duplex+XONE）"
for r in 1 2 3 4 5; do
  run2 "hoist_$r"  TRSM_XONE=0
  run2 "dupb_$r"   TRSM_XONE=0 TRSM_ABHOIST=0
  run2 "dupx_$r"   TRSM_XONE=1 TRSM_ABHOIST=0
done
echo "=== TC1/TC3 守护（默认=v61 形态）"
env OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./trsm_xt2 512 19968 3 2>&1 | grep -E " x |PASS|FAIL"
env OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./trsm_xt2 17024 512 3 2>&1 | grep -E " x |PASS|FAIL"
rm -f trsm_xt2 probe/refxt.o probe/gen_xt
echo "=== DONE $(date +%H:%M:%S)"
