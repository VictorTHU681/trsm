#!/usr/bin/env bash
# v13：NT 只用于 GEMM A 条（对角条用 ld1d）；case1 试 ANT=0 / 不同 mb,kc；三用例默认 + 泛化
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST"; hostname; date
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
gcc $FLAGS bench_trsm.c trsm.c -o trsm_test_v13 -lm -lkblas 2>&1 | grep -E "error"; [[ -x trsm_test_v13 ]] || { echo COMPILE_FAIL; exit 1; }
run() {
    local desc="$1"; shift; local cases="$1"; shift
    echo "--- [$desc]"
    for args in $cases; do
        env "$@" TRSM_PROFILE=2 numactl -N $node $MEMOPT ./trsm_test_v13 ${args//_/ } 2>&1 | grep -E "PASS|FAIL|prof\]  |rror|fault|illed" | tail -3
    done
}
ALL="512_19968_1 2432_17024_1 17024_512_1"
C1="512_19968_1"
run "v13 默认" "$ALL"
run "case1 ant=0" "$C1" TRSM_ANT=0
run "case1 mb256 kc128" "$C1" TRSM_MB=256 TRSM_KC=128
run "case1 mb256 kc128 ant=0" "$C1" TRSM_MB=256 TRSM_KC=128 TRSM_ANT=0
run "case1 mb256 kc64" "$C1" TRSM_MB=256 TRSM_KC=64
run "case1 mb512 nc64 kc64" "$C1" TRSM_MB=512 TRSM_NC=64 TRSM_KC=64
run "case1 mb512 nc32 kc64" "$C1" TRSM_MB=512 TRSM_NC=32 TRSM_KC=64
run "case1 mb128 kc64" "$C1" TRSM_MB=128 TRSM_KC=64
run "case1 mb192 nc64 kc64" "$C1" TRSM_NC=64
run "case1 nosplit" "$C1" TRSM_SPLITF=0
run "case1 pf=0" "$C1" TRSM_PF=0
run "case2/3 ant=0" "2432_17024_1 17024_512_1" TRSM_ANT=0
echo "=== 泛化正确性"
gcc -O3 -fopenmp -DTRSM_NO_SME_ASM -Dl_trsm=l_trsm_ref -c trsm.c -o probe/ref.o 2>&1 | grep -E "error"
gcc $FLAGS probe/gen_test.c trsm.c probe/ref.o -o probe/gen_test -lm 2>&1 | grep -E "error"
timeout 900 numactl -N $node ./probe/gen_test 60 2>&1 | tail -1
TRSM_ANT=0 TRSM_MB=64 timeout 600 numactl -N $node ./probe/gen_test 30 2>&1 | tail -1
echo "=== DONE"; date
