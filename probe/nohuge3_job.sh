#!/usr/bin/env bash
# nohuge3：闭环重测（rh1 线去 SMAPS 仪器毒）——defrag=never 下重缺页能否拿到大页取决于伙伴系统现存 2MiB 块。
#   打印 HBM 节点 buddyinfo（order-9+ 空闲块数），再跑 TC1 三线（base / x4k+rh0 / x4k+rh1）：
#   若 order-9+ 充足且 rh1 恢复到 base → 本地完整演示"页税+回收"闭环；若 order-9 枯竭且 rh1≈rh0 → 复现 cn20489 结论。
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S) $(git log --oneline -1)"
. probe/env.sh
HBM="${MEMOPT#--preferred=}"
echo "--- THP 状态"; cat /sys/kernel/mm/transparent_hugepage/enabled /sys/kernel/mm/transparent_hugepage/defrag 2>/dev/null
echo "--- buddyinfo（HBM 节点 $HBM 与本地节点 1，各 order 空闲块，最后 2 列 = order 9/10 即 2MiB/4MiB）"
grep -E "Node (1|$HBM)," /proc/buddyinfo
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
rm -f fb_nh2; gcc $FLAGS probe/fastbench.c trsm.c -o fb_nh2 -lm 2>&1 | grep -E "rror" | head -3; [[ -x fb_nh2 ]] || { echo COMPILE_FAIL; exit 1; }
run1(){ local lbl="$1"; shift; echo "--- $lbl"; env "$@" OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./fb_nh2 -r 4 512 19968 2>&1 | grep -vE "^  rep"; }
echo "=== TC1 三线交错 x4"
for r in 1 2 3 4; do
  run1 "tc1_base_$r"
  run1 "tc1_x4k_rh0_$r" FB_NOHUGE=x TRSM_REHUGE=0
  run1 "tc1_x4k_rh1_$r" FB_NOHUGE=x 
done
echo "--- 跑后 buddyinfo"
grep -E "Node (1|$HBM)," /proc/buddyinfo
rm -f fb_nh2
echo "=== DONE $(date +%H:%M:%S)"
