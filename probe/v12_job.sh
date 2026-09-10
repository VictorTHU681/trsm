#!/usr/bin/env bash
# v12：X 回写托管进后续 K 循环（SWHOST）+ 对角阶段预占下一任务（RESERVE）；对照与泛化正确性
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST"; hostname; date
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
gcc $FLAGS bench_trsm.c trsm.c -o trsm_test_v12 -lm -lkblas 2>&1 | grep -E "error"; [[ -x trsm_test_v12 ]] || { echo COMPILE_FAIL; exit 1; }
run3() {
    local desc="$1"; shift
    echo "--- [$desc]"
    for args in "512 19968 1" "2432 17024 1" "17024 512 1"; do
        env "$@" TRSM_PROFILE=2 numactl -N $node $MEMOPT ./trsm_test_v12 $args 2>&1 | grep -E "PASS|FAIL|prof\]  |rror|fault|illed" | tail -2
    done
}
run3 "v12 默认（swhost=1 reserve=1）"
run3 "swhost=0 reserve=0（= v10）" TRSM_SWHOST=0 TRSM_RESERVE=0
run3 "swhost=1 reserve=0" TRSM_RESERVE=0
run3 "swhost=0 reserve=1" TRSM_SWHOST=0
run3 "默认 + xpnt=1" TRSM_XPNT=1
run3 "默认 mb256 kc64" TRSM_MB=256
run3 "默认 mb256 kc128" TRSM_MB=256 TRSM_KC=128
echo "=== 泛化正确性"
gcc -O3 -fopenmp -DTRSM_NO_SME_ASM -Dl_trsm=l_trsm_ref -c trsm.c -o probe/ref.o 2>&1 | grep -E "error"
gcc $FLAGS probe/gen_test.c trsm.c probe/ref.o -o probe/gen_test -lm 2>&1 | grep -E "error"
timeout 900 numactl -N $node ./probe/gen_test 60 2>&1 | tail -1
TRSM_MB=64 timeout 600 numactl -N $node ./probe/gen_test 30 2>&1 | tail -1
TRSM_SPLITF=2 TRSM_SPLIT_MINCH=2 timeout 600 numactl -N $node ./probe/gen_test 30 2>&1 | tail -1
echo "=== DONE"; date
