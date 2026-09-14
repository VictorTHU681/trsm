#!/usr/bin/env bash
# xn99：t=末瓦片无 prfm 实例（2a 重试形态）vs fd92af2 基线——TC1 ABBA x6 + gen + kloop0 分桶一览
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S) $(git log --oneline -1)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
git show fd92af2:trsm.c > /tmp/trsm_b99_$$.c
rm -f trsm_xn99 trsm_b99
gcc $FLAGS bench_trsm.c trsm.c -o trsm_xn99 -lm -lkblas 2>&1 | grep -E "rror" | head -5
gcc $FLAGS bench_trsm.c /tmp/trsm_b99_$$.c -o trsm_b99 -lm -lkblas 2>&1 | grep -E "rror" | head -3
[[ -x trsm_xn99 && -x trsm_b99 ]] || { echo "BUILD FAILED"; exit 1; }
echo "=== gen"
gcc -O3 -fopenmp -DTRSM_NO_SME_ASM -Dl_trsm=l_trsm_ref -c trsm.c -o probe/ref99.o 2>&1 | grep -E "error" || true
gcc $FLAGS probe/gen_test.c trsm.c probe/ref99.o -o probe/gen99 -lm 2>&1 | grep -E "error" || true
echo "[owner+steal30]"; TRSM_OWNER=1 TRSM_STEAL=1 timeout 700 numactl -N 1 ./probe/gen99 30 2>&1 | tail -1
echo "[默认20]"; timeout 600 numactl -N 1 ./probe/gen99 20 2>&1 | tail -1
run1(){ local lbl="$1" bin="$2"; echo "--- $lbl"; OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./$bin 512 19968 4 2>&1 | grep -E " x |PASS|FAIL"; }
echo "=== TC1 ABBA x6"
for r in 1 2 3 4 5 6; do
  run1 "new_a$r" trsm_xn99
  run1 "old_a$r" trsm_b99
  run1 "old_b$r" trsm_b99
  run1 "new_b$r" trsm_xn99
done
echo "=== kloop0 分桶（new，PROFILE=2 x2）"
for r in 1 2; do
  TRSM_PROFILE=2 OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./trsm_xn99 512 19968 4 2>&1 | grep -E "每线程平均"
done
echo "=== TC2/TC3 冒烟"
OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./trsm_xn99 2432 17024 2 2>&1 | grep -E " x |PASS|FAIL"
OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./trsm_xn99 17024 512 2 2>&1 | grep -E " x |PASS|FAIL"
rm -f trsm_xn99 trsm_b99 /tmp/trsm_b99_$$.c probe/ref99.o probe/gen99
echo "=== DONE $(date +%H:%M:%S)"
