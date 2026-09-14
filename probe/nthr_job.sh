#!/usr/bin/env bash
# nthr：留空核假说的成本侧（本地干净节点）——37/36 线程 vs 38 的算力损失
#   背景：判题机 loadavg 89-128，38/38 满绑时任何系统线程都会抢占我们（owner 静态调度零容忍，TC1 单次 2ms 调用被抢 0.5ms = +26%）。
#   留 1-2 核给系统 = 抢占免疫，代价 = 本作业实测。若 TC1 代价 <1.5%，线上期望可能为正（TC1 波动带 ±4%）。
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S) $(git log --oneline -1)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
rm -f trsm_nt
gcc $FLAGS bench_trsm.c trsm.c -o trsm_nt -lm -lkblas 2>&1 | grep -E "rror" | head -3
[[ -x trsm_nt ]] || { echo "BUILD FAILED"; exit 1; }
run1(){ local lbl="$1" nt="$2"; shift 2; echo "--- $lbl"; env "$@" OMP_NUM_THREADS=$nt numactl -N 1 $MEMOPT ./trsm_nt ${ARGS} 2>&1 | grep -E " x |PASS|FAIL"; }
echo "=== TC1 三线交错 x6"
for r in 1 2 3 4 5 6; do
  ARGS="512 19968 5" run1 "tc1_38_$r" 38
  ARGS="512 19968 5" run1 "tc1_37_$r" 37
  ARGS="512 19968 5" run1 "tc1_36_$r" 36
done
echo "=== TC2/TC3 三线交错 x2"
for r in 1 2; do
  for args in "2432 17024 3" "17024 512 3"; do
    ARGS="$args" run1 "tc_${args%% *}_38_$r" 38
    ARGS="$args" run1 "tc_${args%% *}_37_$r" 37
    ARGS="$args" run1 "tc_${args%% *}_36_$r" 36
  done
done
rm -f trsm_nt
echo "=== DONE $(date +%H:%M:%S)"
