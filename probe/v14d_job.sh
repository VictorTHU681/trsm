#!/usr/bin/env bash
# v14d：托管回写源预取距离 0 / 1K / 2K / 4K
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST"; hostname; date
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
for d in 0 1024 2048 4096; do gcc $FLAGS -DTRSM_SW_PFD=$d bench_trsm.c trsm.c -o tt_pfd$d -lm -lkblas 2>&1 | grep -E "error"; done
run() {
    local bin="$1" desc="$2"; shift 2
    echo "--- [$bin $desc]"
    for args in "512 19968 1" "2432 17024 1" "17024 512 1"; do
        env "$@" TRSM_PROFILE=2 numactl -N $node $MEMOPT ./$bin $args 2>&1 | grep -E "PASS|FAIL|prof\]  |rror|fault|illed" | tail -3
    done
}
for d in 0 1024 2048 4096; do run tt_pfd$d "源预取 $d"; done
run tt_pfd2048 "2048 第二轮"
run tt_pfd0 "swhost=0" TRSM_SWHOST=0
echo "=== DONE"; date
