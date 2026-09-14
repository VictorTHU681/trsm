#!/usr/bin/env bash
# v64f：v3 回滚后终验——TC1 6v6 ABBA + TC2/TC3 2v2 + gen 核心组合
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S) $(git log --oneline -1)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
git show dca36c8:trsm.c > /tmp/trsm_v63f_$$.c
rm -f trsm_v64f trsm_v63f
gcc $FLAGS bench_trsm.c trsm.c -o trsm_v64f -lm -lkblas 2>&1 | grep -E "rror" | head -3
gcc $FLAGS bench_trsm.c /tmp/trsm_v63f_$$.c -o trsm_v63f -lm -lkblas 2>&1 | grep -E "rror" | head -3
[[ -x trsm_v64f && -x trsm_v63f ]] || { echo "BUILD FAILED"; exit 1; }
echo "=== gen"
gcc -O3 -fopenmp -DTRSM_NO_SME_ASM -Dl_trsm=l_trsm_ref -c trsm.c -o probe/reff4.o 2>&1 | grep -E "error" || true
gcc $FLAGS probe/gen_test.c trsm.c probe/reff4.o -o probe/genf4 -lm 2>&1 | grep -E "error" || true
echo "[默认30]";  timeout 700 numactl -N 1 ./probe/genf4 30 2>&1 | tail -1
echo "[owner+steal20]"; TRSM_OWNER=1 TRSM_STEAL=1 timeout 600 numactl -N 1 ./probe/genf4 20 2>&1 | tail -1
run3(){ local lbl="$1" bin="$2" args="$3"; echo "--- $lbl"; OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./$bin $args 2>&1 | grep -E " x |PASS|FAIL"; }
for r in 1 2 3; do
  run3 "tc1_new_a$r" trsm_v64f "512 19968 4"
  run3 "tc1_old_a$r" trsm_v63f "512 19968 4"
  run3 "tc1_old_b$r" trsm_v63f "512 19968 4"
  run3 "tc1_new_b$r" trsm_v64f "512 19968 4"
done
for r in 1 2; do
  run3 "tc2_new_$r" trsm_v64f "2432 17024 4"
  run3 "tc2_old_$r" trsm_v63f "2432 17024 4"
  run3 "tc3_new_$r" trsm_v64f "17024 512 4"
  run3 "tc3_old_$r" trsm_v63f "17024 512 4"
done
rm -f trsm_v64f trsm_v63f /tmp/trsm_v63f_$$.c probe/reff4.o probe/genf4
echo "=== DONE $(date +%H:%M:%S)"
