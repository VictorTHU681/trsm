#!/usr/bin/env bash
# dx1：DDX 生产筛查——gen + TC2 同二进制 A/B x6 + TC1/TC3 守护
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S) $(git log --oneline -1)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
rm -f trsm_dx1
gcc $FLAGS bench_trsm.c trsm.c -o trsm_dx1 -lm -lkblas 2>&1 | grep -E "rror" | head -5
[[ -x trsm_dx1 ]] || { echo "BUILD FAILED"; exit 1; }
echo "=== gen"
gcc -O3 -fopenmp -DTRSM_NO_SME_ASM -Dl_trsm=l_trsm_ref -c trsm.c -o probe/refx1.o 2>&1 | grep -E "error" || true
gcc $FLAGS probe/gen_test.c trsm.c probe/refx1.o -o probe/genx1 -lm 2>&1 | grep -E "error" || true
echo "[默认20]";  timeout 600 numactl -N 1 ./probe/genx1 20 2>&1 | tail -1
echo "[DDX=1 30]"; TRSM_DDX=1 timeout 700 numactl -N 1 ./probe/genx1 30 2>&1 | tail -1
run2(){ local lbl="$1"; shift; echo "--- $lbl"; env "$@" TRSM_PROFILE=2 OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./trsm_dx1 2432 17024 4 2>&1 | grep -E " x |PASS|FAIL|每线程平均"; }
echo "=== TC2 ABBA x6"
for r in 1 2 3 4 5 6; do
  run2 "base_$r"
  run2 "ddx_$r"  TRSM_DDX=1
  run2 "ddx_b$r" TRSM_DDX=1
  run2 "base_b$r"
done
echo "=== TC1/TC3 守护（DDX=1 应无路径变化）"
TRSM_DDX=1 OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./trsm_dx1 512 19968 3 2>&1 | grep -E " x |PASS|FAIL"
TRSM_DDX=1 OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./trsm_dx1 17024 512 3 2>&1 | grep -E " x |PASS|FAIL"
rm -f trsm_dx1 probe/refx1.o probe/genx1
echo "=== DONE $(date +%H:%M:%S)"
