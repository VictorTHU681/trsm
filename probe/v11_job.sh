#!/usr/bin/env bash
# v11 试验：流式操作数不滞留 L2 的手段（ldnt1d A/B、pldl1strm 预取）× 更大 kc/mb
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST"; hostname; date
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
build() { gcc $FLAGS $2 bench_trsm.c trsm.c -o "$1" -lm -lkblas 2>&1 | grep -E "error"; [[ -x "$1" ]] || { echo "COMPILE_FAIL $1"; exit 1; }; }
build tt_base ""
build tt_ant "-DTRSM_LDA_NT=1"
build tt_abnt "-DTRSM_LDA_NT=1 -DTRSM_LDB_NT=1"
build tt_strm "-DTRSM_PF_HINT=pldl1strm"
build tt_ant_strm "-DTRSM_LDA_NT=1 -DTRSM_PF_HINT=pldl1strm"
run2() {
    local bin="$1" desc="$2"; shift 2
    echo "--- [$bin $desc]"
    for args in "2432 17024 1" "17024 512 1"; do
        env "$@" TRSM_PROFILE=2 numactl -N $node $MEMOPT ./$bin $args 2>&1 | grep -E "PASS|FAIL|prof\]  |rror|fault|illed" | tail -2
    done
}
for bin in tt_base tt_ant tt_abnt tt_strm tt_ant_strm; do
    run2 $bin "mb192 kc64"  TRSM_MB=192 TRSM_KC=64
    run2 $bin "mb192 kc128" TRSM_MB=192 TRSM_KC=128   # kc|mb 不成立 → 自动降到 96
    run2 $bin "mb256 kc128" TRSM_MB=256 TRSM_KC=128
    run2 $bin "mb256 kc64"  TRSM_MB=256 TRSM_KC=64
done
echo "=== DONE"; date
