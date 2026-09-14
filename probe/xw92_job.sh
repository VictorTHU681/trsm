#!/usr/bin/env bash
# xw92：补全因子矩阵——dwpf0 单独 / xpw+dpf0 / xpwd+dpf0（全开）/ pch+dwpf0，找 TC2 最优组合；附赢家 TC1/TC3 冒烟
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S) $(git log --oneline -1)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
rm -f trsm_xw92
gcc $FLAGS bench_trsm.c trsm.c -o trsm_xw92 -lm -lkblas 2>&1 | grep -E "rror" | head -5
[[ -x trsm_xw92 ]] || { echo "BUILD FAILED"; exit 1; }
run2(){ local lbl="$1"; shift; echo "--- $lbl"; env "$@" TRSM_PROFILE=2 OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./trsm_xw92 2432 17024 4 2>&1 | grep -E " x |PASS|FAIL|每线程平均"; }
echo "=== TC2 六因子 x4"
for r in 1 2 3 4; do
  run2 "base_$r"
  run2 "dw0_$r"    TRSM_DWPF=0
  run2 "xpwD_$r"   TRSM_XPW=1 TRSM_DPF=0
  run2 "all3_$r"   TRSM_XPW=1 TRSM_DWPF=0 TRSM_DPF=0
  run2 "pchdw_$r"  TRSM_PCHAIN=1 TRSM_DWPF=0
  run2 "xpwd_$r"   TRSM_XPW=1 TRSM_DWPF=0
done
echo "=== TC1/TC3 在 dwpf0/dpf0 下的影响"
for a in "512 19968" "17024 512"; do
  for e in "base" "dw0 TRSM_DWPF=0" "dpf0 TRSM_DPF=0"; do
    set -- $e; lbl=$1; shift
    echo "--- tc_${a%% *}_$lbl"
    env "$@" OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./trsm_xw92 $a 3 2>&1 | grep -E " x |PASS|FAIL"
  done
done
rm -f trsm_xw92
echo "=== DONE $(date +%H:%M:%S)"
