#!/usr/bin/env bash
# v64gate：simp 三件套产品化后的完整门禁——gen 全组合 + 三用例 新默认 vs v63(dca36c8) ABBA x4
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S) $(git log --oneline -1)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
git show dca36c8:trsm.c > /tmp/trsm_v63g_$$.c
rm -f trsm_v64n trsm_v63o
gcc $FLAGS bench_trsm.c trsm.c -o trsm_v64n -lm -lkblas 2>&1 | grep -E "rror" | head -3
gcc $FLAGS bench_trsm.c /tmp/trsm_v63g_$$.c -o trsm_v63o -lm -lkblas 2>&1 | grep -E "rror" | head -3
[[ -x trsm_v64n && -x trsm_v63o ]] || { echo "BUILD FAILED"; exit 1; }
echo "=== gen 全组合"
gcc -O3 -fopenmp -DTRSM_NO_SME_ASM -Dl_trsm=l_trsm_ref -c trsm.c -o probe/refg4.o 2>&1 | grep -E "error" || true
gcc $FLAGS probe/gen_test.c trsm.c probe/refg4.o -o probe/geng4 -lm 2>&1 | grep -E "error" || true
echo "[默认30]";  timeout 700 numactl -N 1 ./probe/geng4 30 2>&1 | tail -1
echo "[owner20]"; TRSM_OWNER=1 timeout 600 numactl -N 1 ./probe/geng4 20 2>&1 | tail -1
echo "[steal20]"; TRSM_OWNER=1 TRSM_STEAL=1 timeout 600 numactl -N 1 ./probe/geng4 20 2>&1 | tail -1
echo "[cfold20]"; TRSM_OWNER=1 TRSM_CFOLD=1 timeout 600 numactl -N 1 ./probe/geng4 20 2>&1 | tail -1
echo "[xpw20]";   TRSM_XPW=1 timeout 600 numactl -N 1 ./probe/geng4 20 2>&1 | tail -1
echo "[5thr20]";  OMP_NUM_THREADS=5 timeout 600 numactl -N 1 ./probe/geng4 20 2>&1 | tail -1
run3(){ local lbl="$1" bin="$2" args="$3"; echo "--- $lbl"; OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./$bin $args 2>&1 | grep -E " x |PASS|FAIL"; }
for tc in "tc1 512 19968 4" "tc2 2432 17024 4" "tc3 17024 512 4"; do
  set -- $tc; nm=$1; shift; args="$*"
  for r in 1 2; do
    run3 "${nm}_new_a$r" trsm_v64n "$args"
    run3 "${nm}_old_a$r" trsm_v63o "$args"
    run3 "${nm}_old_b$r" trsm_v63o "$args"
    run3 "${nm}_new_b$r" trsm_v64n "$args"
  done
done
rm -f trsm_v64n trsm_v63o /tmp/trsm_v63g_$$.c probe/refg4.o probe/geng4
echo "=== DONE $(date +%H:%M:%S)"
