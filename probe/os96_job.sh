#!/usr/bin/env bash
# os96：OSW+ 筛查（守卫放宽 + NEXTA 补发）——TC1 三线 ABBA x5 + gen
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S) $(git log --oneline -1)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
rm -f trsm_os96
gcc $FLAGS bench_trsm.c trsm.c -o trsm_os96 -lm -lkblas 2>&1 | grep -E "rror" | head -3
[[ -x trsm_os96 ]] || { echo "BUILD FAILED"; exit 1; }
echo "=== gen"
gcc -O3 -fopenmp -DTRSM_NO_SME_ASM -Dl_trsm=l_trsm_ref -c trsm.c -o probe/ref96.o 2>&1 | grep -E "error" || true
gcc $FLAGS probe/gen_test.c trsm.c probe/ref96.o -o probe/gen96 -lm 2>&1 | grep -E "error" || true
echo "[gen OSW owner30]"; TRSM_OSW=1 TRSM_OWNER=1 timeout 700 numactl -N 1 ./probe/gen96 30 2>&1 | tail -1
echo "[gen OSW steal20]"; TRSM_OSW=1 TRSM_OWNER=1 TRSM_STEAL=1 timeout 600 numactl -N 1 ./probe/gen96 20 2>&1 | tail -1
run1(){ local lbl="$1"; shift; echo "--- $lbl"; env "$@" TRSM_PROFILE=2 OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./trsm_os96 512 19968 4 2>&1 | grep -E " x |PASS|FAIL|每线程平均"; }
for r in 1 2 3 4 5; do
  run1 "base_$r"
  run1 "osw_$r"   TRSM_OSW=1
  run1 "oswd_$r"  TRSM_OSW=1 TRSM_DWPF=0
  run1 "oswd_b$r" TRSM_OSW=1 TRSM_DWPF=0
  run1 "osw_b$r"  TRSM_OSW=1
  run1 "base_b$r"
done
rm -f trsm_os96 probe/ref96.o probe/gen96
echo "=== DONE $(date +%H:%M:%S)"
