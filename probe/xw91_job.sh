#!/usr/bin/env bash
# xw91：XPW（EV 双写 Xp）+ PCHAIN + DPF 因子筛查（Fable-TC2 候选 A/B）
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S) $(git log --oneline -1)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
rm -f trsm_xw91
gcc $FLAGS bench_trsm.c trsm.c -o trsm_xw91 -lm -lkblas 2>&1 | grep -E "rror" | head -5
[[ -x trsm_xw91 ]] || { echo "BUILD FAILED"; exit 1; }
echo "=== gen"
gcc -O3 -fopenmp -DTRSM_NO_SME_ASM -Dl_trsm=l_trsm_ref -c trsm.c -o probe/refw9.o 2>&1 | grep -E "error" || true
gcc $FLAGS probe/gen_test.c trsm.c probe/refw9.o -o probe/genw9 -lm 2>&1 | grep -E "error" || true
echo "[gen XPW 默认30]"; TRSM_XPW=1 timeout 700 numactl -N 1 ./probe/genw9 30 2>&1 | tail -1
echo "[gen XPW+PCHAIN owner20]"; TRSM_XPW=1 TRSM_OWNER=1 timeout 600 numactl -N 1 ./probe/genw9 20 2>&1 | tail -1
run2(){ local lbl="$1"; shift; echo "--- $lbl"; env "$@" TRSM_PROFILE=2 OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./trsm_xw91 2432 17024 4 2>&1 | grep -E " x |PASS|FAIL|每线程平均"; }
echo "=== TC2 五因子 x4"
for r in 1 2 3 4; do
  run2 "base_$r"
  run2 "xpw_$r"   TRSM_XPW=1
  run2 "pch_$r"   TRSM_PCHAIN=1
  run2 "xpwd_$r"  TRSM_XPW=1 TRSM_DWPF=0
  run2 "dpf0_$r"  TRSM_DPF=0
done
echo "=== TC1/TC3 冒烟（XPW 开，应无路径变化）"
TRSM_XPW=1 OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./trsm_xw91 512 19968 3 2>&1 | grep -E " x |PASS|FAIL"
TRSM_XPW=1 OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./trsm_xw91 17024 512 3 2>&1 | grep -E " x |PASS|FAIL"
rm -f trsm_xw91 probe/refw9.o probe/genw9
echo "=== DONE $(date +%H:%M:%S)"
