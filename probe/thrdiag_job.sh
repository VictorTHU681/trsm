#!/usr/bin/env bash
# TC1 慢线程诊断：TRSM_PROFILE=4 每线程分相（含 khead、span、resid、尾部 idle），
# 跨进程重复 ×2 + owner 边界扰动（C0）×2 + 显式任务表对照，定位"慢线程绑定区间形状"的时间去向。
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S) $(git log --oneline -1)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
rm -f trsm_test
gcc $FLAGS bench_trsm.c trsm.c -o trsm_test -lm -lkblas 2>&1 | grep -E "rror" | head -5 || true
[ -x ./trsm_test ] || { echo "BUILD FAILED"; exit 1; }
run1() {  # $1=label, 其余=额外环境变量
  local lbl="$1"; shift
  echo "--- ROUND $lbl"
  env "$@" TRSM_PROFILE=4 OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./trsm_test 512 19968 7 2>&1
}
run1 base1 X=0
run1 base2 X=0
run1 c0_160 TRSM_OWNER_C0=160
run1 c0_240 TRSM_OWNER_C0=240
run1 expl_owner TRSM_IMPLICIT_OWNER=0
echo "=== DONE $(date +%H:%M:%S)"
