#!/usr/bin/env bash
# pk1：P0（PACKNT）筛查——探针三合一 + 生产 TC3 同二进制 A/B x6 + gen
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S) $(git log --oneline -1)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
rm -f trsm_pk1 probe/packnt_test
gcc $FLAGS bench_trsm.c trsm.c -o trsm_pk1 -lm -lkblas 2>&1 | grep -E "rror" | head -5
gcc -O2 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp probe/packnt_test.c -o probe/packnt_test -lm 2>&1 | grep -E "rror" | head -5
[[ -x trsm_pk1 && -x probe/packnt_test ]] || { echo "BUILD FAILED"; exit 1; }
echo "=== 探针（单核 DDR 放置）"
timeout 900 numactl -N 1 taskset -c 40 ./probe/packnt_test 2>&1
echo "=== gen"
gcc -O3 -fopenmp -DTRSM_NO_SME_ASM -Dl_trsm=l_trsm_ref -c trsm.c -o probe/refp1.o 2>&1 | grep -E "error" || true
gcc $FLAGS probe/gen_test.c trsm.c probe/refp1.o -o probe/genp1 -lm 2>&1 | grep -E "error" || true
echo "[默认20]";   timeout 600 numactl -N 1 ./probe/genp1 20 2>&1 | tail -1
echo "[PACKNT20]"; TRSM_PACKNT=1 timeout 600 numactl -N 1 ./probe/genp1 20 2>&1 | tail -1
run3(){ local lbl="$1"; shift; echo "--- $lbl"; env "$@" TRSM_PROFILE=2 OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./trsm_pk1 17024 512 4 2>&1 | grep -E " x |PASS|FAIL|每线程平均" ; }
echo "=== TC3 ABBA x6"
for r in 1 2 3 4 5 6; do
  run3 "base_$r"
  run3 "pnt_$r"  TRSM_PACKNT=1
  run3 "pnt_b$r" TRSM_PACKNT=1
  run3 "base_b$r"
done
echo "=== TC1/TC2 冒烟（PACKNT=1 应无路径变化）"
TRSM_PACKNT=1 OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./trsm_pk1 512 19968 2 2>&1 | grep -E " x |PASS|FAIL"
TRSM_PACKNT=1 OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./trsm_pk1 2432 17024 2 2>&1 | grep -E " x |PASS|FAIL"
rm -f trsm_pk1 probe/refp1.o probe/genp1 probe/packnt_test
echo "=== DONE $(date +%H:%M:%S)"
