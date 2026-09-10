#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST"; hostname; date
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
gcc $FLAGS bench_trsm.c trsm.c -o trsm_test_v12c -lm -lkblas 2>&1 | grep -E "error"; [[ -x trsm_test_v12c ]] || { echo COMPILE_FAIL; exit 1; }
run3() {
    local desc="$1"; shift
    echo "--- [$desc]"
    for args in "512 19968 1" "2432 17024 1" "17024 512 1"; do
        env "$@" TRSM_PROFILE=2 numactl -N $node $MEMOPT ./trsm_test_v12c $args 2>&1 | grep -E "PASS|FAIL|prof\]  |rror|fault|illed" | tail -3
    done
}
run3 "v12c 默认"
run3 "xpnt=1" TRSM_XPNT=1
echo "=== DONE"; date
