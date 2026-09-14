#!/usr/bin/env bash
# dx70：DXW 对角双写筛查（扫雷 P2）——预期 store 0.12→~0.01、ddump 0.055→0.08-0.11，净 −0.05~−0.08ms（+2.5~4%）
#   四格：base / DXW / DXW+DWPF=0 / DWPF=0（DXW 与对角窗游标分池交叉）
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S) $(git log --oneline -1)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
rm -f trsm_dx70
gcc $FLAGS bench_trsm.c trsm.c -o trsm_dx70 -lm -lkblas 2>&1 | grep -E "rror" | head -3
[[ -x trsm_dx70 ]] || { echo "BUILD FAILED"; exit 1; }
echo "=== gen 正确性（DXW=1）"
gcc -O3 -fopenmp -DTRSM_NO_SME_ASM -Dl_trsm=l_trsm_ref -c trsm.c -o probe/ref70.o 2>&1 | grep -E "error" || true
gcc $FLAGS probe/gen_test.c trsm.c probe/ref70.o -o probe/gen_test70 -lm 2>&1 | grep -E "error" || true
echo "[gen DXW 强制owner 25]";        TRSM_DXW=1 TRSM_OWNER=1 timeout 600 numactl -N 1 ./probe/gen_test70 25 2>&1 | tail -1
echo "[gen DXW+CFOLD 强制owner 20]";  TRSM_DXW=1 TRSM_OWNER=1 TRSM_CFOLD=1 timeout 600 numactl -N 1 ./probe/gen_test70 20 2>&1 | tail -1
echo "=== TC1 四格交错 x5（官方 bench + PROFILE=2）"
run1(){ local lbl="$1"; shift; echo "--- $lbl"; env "$@" TRSM_PROFILE=2 OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./trsm_dx70 512 19968 5 2>&1 | grep -E " x |PASS|FAIL|每线程平均"; }
for r in 1 2 3 4 5; do
  run1 "base_$r"
  run1 "dxw_$r"    TRSM_DXW=1
  run1 "dxwnd_$r"  TRSM_DXW=1 TRSM_DWPF=0
  run1 "nd_$r"     TRSM_DWPF=0
done
rm -f trsm_dx70 probe/ref70.o probe/gen_test70
echo "=== DONE $(date +%H:%M:%S)"
