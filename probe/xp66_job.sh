#!/usr/bin/env bash
# xp66：XPRE 筛查（审计 §10w 候选 1）——下一任务 X 块借 GEMM 末段白发游标槽铺 L2
#   判读：load 应显著降（满覆盖理论 0.30→~0.10）且 kloop/dkloop 合计涨 ≤0.04 → 候选成立；
#         load 不动 → 槽位/配速失效（对照 dw65 的"内核内配速是唯一活形态"——XPRE 正是内核内配速）
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S) $(git log --oneline -1)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
rm -f trsm_xp66
gcc $FLAGS bench_trsm.c trsm.c -o trsm_xp66 -lm -lkblas 2>&1 | grep -E "rror" | head -3
[[ -x trsm_xp66 ]] || { echo "BUILD FAILED"; exit 1; }
run1(){ local lbl="$1"; shift; echo "--- $lbl"; env "$@" TRSM_PROFILE=2 OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./trsm_xp66 512 19968 5 2>&1 | grep -E " x |PASS|FAIL|每线程平均"; }
echo "=== TC1 四线交错 x5"
for r in 1 2 3 4 5; do
  run1 "off_$r"
  run1 "xp1_$r"   TRSM_XPRE=1
  run1 "xp2_$r"   TRSM_XPRE=2
  run1 "xp2p8_$r" TRSM_XPRE=2 TRSM_XPRE_PAIRS=8
done
rm -f trsm_xp66
echo "=== DONE $(date +%H:%M:%S)"
