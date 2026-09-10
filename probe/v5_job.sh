#!/usr/bin/env bash
# v5 作业：泛化正确性 + 三用例 profile（Xp 面板副本、双走线器）
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST"; hostname; date
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
gcc $FLAGS bench_trsm.c trsm.c -o trsm_test -lm -lkblas 2>&1 | grep -E "error"; [[ -x trsm_test ]] || { echo COMPILE_FAIL; exit 1; }
gcc -O3 -fopenmp -DTRSM_NO_SME_ASM -Dl_trsm=l_trsm_ref -c trsm.c -o probe/ref.o 2>&1 | grep -E "error"
gcc $FLAGS probe/gen_test.c trsm.c probe/ref.o -o probe/gen_test -lm 2>&1 | grep -E "error"; [[ -x probe/gen_test ]] || { echo GEN_COMPILE_FAIL; exit 1; }
echo "=== 泛化正确性"
timeout 600 numactl -N $node ./probe/gen_test 30 2>&1 | tail -1
TRSM_JIT=1 timeout 600 numactl -N $node ./probe/gen_test 30 2>&1 | tail -1
TRSM_JIT=1 TRSM_KC=256 timeout 600 numactl -N $node ./probe/gen_test 20 2>&1 | tail -1
TRSM_JIT=1 TRSM_MB=64 TRSM_NC=128 TRSM_KC=32 timeout 600 numactl -N $node ./probe/gen_test 20 2>&1 | tail -1
TRSM_JIT=1 TRSM_MB=96 TRSM_NC=96 TRSM_KC=96 timeout 600 numactl -N $node ./probe/gen_test 20 2>&1 | tail -1
TRSM_JIT=1 TRSM_SPLITF=40 TRSM_MB=128 TRSM_KC=32 timeout 600 numactl -N $node ./probe/gen_test 20 2>&1 | tail -1
TRSM_SPLITF=40 TRSM_MB=128 TRSM_KC=64 timeout 600 numactl -N $node ./probe/gen_test 20 2>&1 | tail -1
TRSM_JIT=1 TRSM_PACK=0 timeout 600 numactl -N $node ./probe/gen_test 20 2>&1 | tail -1
run3() {
    local desc="$1"; shift
    echo "--- [$desc]"
    for args in "512 19968 1" "2432 17024 1" "17024 512 1"; do
        env "$@" TRSM_PROFILE=2 numactl -N $node $MEMOPT ./trsm_test $args 2>&1 | grep -E "PASS|FAIL|prof\]|Error" | tail -3
    done
}
echo "=== 三用例分相 profile"
run3 "kc128 默认" A=1
run3 "kc256" TRSM_KC=256
run3 "kc128 xp=hbm(默认策略)" TRSM_XPNODE=-1
run3 "kc128 xp=ddr 全部" TRSM_XPNODE=-2
run3 "kc128 splitf=12 maxseg=8" TRSM_SPLITF=12 TRSM_MAXSEG=8
run3 "kc128 mb224" TRSM_MB=224 TRSM_KC=32
run3 "kc128 mb192 kc192" TRSM_MB=192 TRSM_KC=192
run3 "kc128 pf=0" TRSM_PF=0
echo "=== DONE"; date
