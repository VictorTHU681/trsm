#!/usr/bin/env bash
# v14c：托管回写诊断：速率 2 vs 4；源装载热地址（结果 FAIL，只看时序）
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST"; hostname; date
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
gcc $FLAGS bench_trsm.c trsm.c -o tt_r2 -lm -lkblas 2>&1 | grep -E "error"
gcc $FLAGS -DTRSM_SW_RATE=4 bench_trsm.c trsm.c -o tt_r4 -lm -lkblas 2>&1 | grep -E "error"
gcc $FLAGS -DTRSM_SW_DIAG=1 bench_trsm.c trsm.c -o tt_diag -lm -lkblas 2>&1 | grep -E "error"
gcc $FLAGS -DTRSM_SW_DIAG=1 -DTRSM_SW_RATE=4 bench_trsm.c trsm.c -o tt_diag4 -lm -lkblas 2>&1 | grep -E "error"
run() {
    local bin="$1" desc="$2"; shift 2
    echo "--- [$bin $desc]"
    for args in "512 19968 1" "2432 17024 1" "17024 512 1"; do
        env "$@" TRSM_PROFILE=2 numactl -N $node $MEMOPT ./$bin $args 2>&1 | grep -E "PASS|FAIL|prof\]  |rror|fault|illed" | tail -3
    done
}
run tt_r2 "速率 2"
run tt_r4 "速率 4"
run tt_diag "速率 2 源热（FAIL 预期）"
run tt_diag4 "速率 4 源热（FAIL 预期）"
run tt_r2 "swhost=0" TRSM_SWHOST=0
echo "=== DONE"; date
