#!/usr/bin/env bash
# DSPLIT 归档正确性：gen_test 默认 + 强开（mode1/2/4 小形状、残块、奇形状）
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
gcc -O3 -fopenmp -DTRSM_NO_SME_ASM -Dl_trsm=l_trsm_ref -c trsm.c -o probe/ref.o 2>&1 | grep -E "error" | head -5
gcc $FLAGS probe/gen_test.c trsm.c probe/ref.o -o probe/gen_test -lm 2>&1 | grep -E "error" | head -5
[[ -x probe/gen_test ]] || { echo COMPILE_FAIL; exit 1; }
echo "--- 默认（DSPLIT=0）"
timeout 900 numactl -N 1 ./probe/gen_test 40 2>&1 | tail -1
echo "--- DSPLIT=1"
TRSM_DSPLIT=1 timeout 900 numactl -N 1 ./probe/gen_test 40 2>&1 | tail -1
echo "--- DSPLIT=1 + DIRECT=0（mode2/4 语义小形状）"
TRSM_DSPLIT=1 TRSM_DIRECT=0 timeout 900 numactl -N 1 ./probe/gen_test 40 2>&1 | tail -1
echo "--- DSPLIT=1 + JIT_MB=0（流水打包 mode4）"
TRSM_DSPLIT=1 TRSM_JIT_MB=0 TRSM_DIRECT=0 timeout 900 numactl -N 1 ./probe/gen_test 40 2>&1 | tail -1
echo "=== DONE $(date +%H:%M:%S)"
