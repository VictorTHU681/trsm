#!/usr/bin/env bash
# v14b：托管回写 stnt1d vs st1d
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST"; hostname; date
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
gcc $FLAGS bench_trsm.c trsm.c -o tt_swnt -lm -lkblas 2>&1 | grep -E "error"
gcc $FLAGS -DTRSM_SW_NT=0 bench_trsm.c trsm.c -o tt_swst -lm -lkblas 2>&1 | grep -E "error"
run() {
    local bin="$1" desc="$2"; shift 2
    echo "--- [$bin $desc]"
    for args in "512 19968 1" "2432 17024 1" "17024 512 1"; do
        env "$@" TRSM_PROFILE=2 numactl -N $node $MEMOPT ./$bin $args 2>&1 | grep -E "PASS|FAIL|prof\]  |rror|fault|illed" | tail -3
    done
}
run tt_swst "st1d 托管"
run tt_swnt "stnt1d 托管"
run tt_swst "st1d 托管 第二轮"
run tt_swnt "swhost=0" TRSM_SWHOST=0
echo "=== DONE"; date
