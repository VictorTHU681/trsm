#!/usr/bin/env bash
# dw65：对角窗预取形态筛查（tc1p59 消融证实游标版 pldl1keep 是死的；本作业测 L2 突发替代形态）
#   四线：dwpf1=现状游标 / dwpf0=关 / dwpf2=瓦片间 pldl2keep 突发同配速(~216行) / dwpf3=加密(~432行)
#   判读：dwpf2/3 若 load 降 ≥0.03ms 且 dkloop 不涨、task 净降 → 候选；全平 → 对角窗也关闭，#3 终局
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S) $(git log --oneline -1)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
rm -f trsm_dw65
gcc $FLAGS bench_trsm.c trsm.c -o trsm_dw65 -lm -lkblas 2>&1 | grep -E "rror" | head -3
[[ -x trsm_dw65 ]] || { echo "BUILD FAILED"; exit 1; }
run1(){ local lbl="$1"; shift; echo "--- $lbl"; env "$@" TRSM_PROFILE=2 OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./trsm_dw65 512 19968 5 2>&1 | grep -E " x |PASS|FAIL|每线程平均"; }
echo "=== TC1 四线交错 x5"
for r in 1 2 3 4 5; do
  run1 "dw1_$r"
  run1 "dw0_$r" TRSM_DWPF=0
  run1 "dw2_$r" TRSM_DWPF=2
  run1 "dw3_$r" TRSM_DWPF=3
done
rm -f trsm_dw65
echo "=== DONE $(date +%H:%M:%S)"
