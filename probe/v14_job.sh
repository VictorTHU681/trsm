#!/usr/bin/env bash
# v14：托管回写放宽对齐条件（bench 的 B 只 16 B 对齐）；小 m 默认 ant=0
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST"; hostname; date
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
gcc $FLAGS bench_trsm.c trsm.c -o trsm_test_v14 -lm -lkblas 2>&1 | grep -E "error"; [[ -x trsm_test_v14 ]] || { echo COMPILE_FAIL; exit 1; }
run() {
    local desc="$1"; shift; local cases="$1"; shift
    echo "--- [$desc]"
    for args in $cases; do
        env "$@" TRSM_PROFILE=2 numactl -N $node $MEMOPT ./trsm_test_v14 ${args//_/ } 2>&1 | grep -E "PASS|FAIL|prof\]  |rror|fault|illed" | tail -3
    done
}
ALL="512_19968_1 2432_17024_1 17024_512_1"
run "v14 默认" "$ALL"
run "swhost=0" "$ALL" TRSM_SWHOST=0
run "v14 默认 第二轮" "$ALL"
echo "=== 泛化正确性"
gcc -O3 -fopenmp -DTRSM_NO_SME_ASM -Dl_trsm=l_trsm_ref -c trsm.c -o probe/ref.o 2>&1 | grep -E "error"
gcc $FLAGS probe/gen_test.c trsm.c probe/ref.o -o probe/gen_test -lm 2>&1 | grep -E "error"
timeout 900 numactl -N $node ./probe/gen_test 60 2>&1 | tail -1
TRSM_MB=64 timeout 600 numactl -N $node ./probe/gen_test 30 2>&1 | tail -1
echo "=== DONE"; date
