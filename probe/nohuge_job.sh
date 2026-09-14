#!/usr/bin/env bash
# nohuge：本地页税仿真（REHUGE 审计 Lane B 方案）——FB_NOHUGE 首触前强制 4K 页，本机 THP=always 下复现判题机低覆盖态
# 判读标准：
#   tc1_x4k_rh0 vs tc1_base 差值 = X 页税@1.55GHz（若 −3~5% → 判题机 REHUGE +6.5% 有机制支撑；≈0 → 收益主体是环境抖动）
#   tc1_x4k_rh1 应回到 tc1_base（REHUGE 的 MADV_HUGEPAGE 清 NOHUGEPAGE 标志、重铺生效的直接证明）
#   tc3_l4k vs tc3_lbase 差值 = REHUGE-L 假想收益的绝对上界（<0.5% 则永久关闭 REHUGE-L 方向）
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S) $(git log --oneline -1)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
rm -f fb_nh; gcc $FLAGS probe/fastbench.c trsm.c -o fb_nh -lm 2>&1 | grep -E "rror" | head -5; [[ -x fb_nh ]] || { echo COMPILE_FAIL; exit 1; }
run1() { local lbl="$1"; shift; echo "--- $lbl"; env "$@" OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./fb_nh ${ARGS} 2>&1 | grep -vE "^  rep"; }

echo "=== TC1 X 页税三线交错 x6（base / x4k+REHUGE0 / x4k+REHUGE1）"
for r in 1 2 3 4 5 6; do
  ARGS="-r 4 512 19968" run1 "tc1_base_$r"
  ARGS="-r 4 512 19968" run1 "tc1_x4k_rh0_$r" FB_NOHUGE=x TRSM_REHUGE=0
  ARGS="-r 4 512 19968" run1 "tc1_x4k_rh1_$r" FB_NOHUGE=x
done

echo "=== TC2/TC3 X 页税两线交错 x2"
for r in 1 2; do
  ARGS="-r 3 2432 17024 17024 512" run1 "tc23_base_$r"
  ARGS="-r 3 2432 17024 17024 512" run1 "tc23_x4k_rh0_$r" FB_NOHUGE=x TRSM_REHUGE=0
done

echo "=== TC3 L 页税（REHUGE-L 绝对上界）两线交错 x4"
for r in 1 2 3 4; do
  ARGS="-r 3 17024 512" run1 "tc3_lbase_$r"
  ARGS="-r 3 17024 512" run1 "tc3_l4k_$r" FB_NOHUGE=l
done
echo "=== DONE $(date +%H:%M:%S)"
