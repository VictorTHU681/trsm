#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST"; hostname; date
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
gcc $FLAGS bench_trsm.c trsm.c -o trsm_test_c3 -lm -lkblas 2>&1 | grep -E "error"; [[ -x trsm_test_c3 ]] || { echo COMPILE_FAIL; exit 1; }
run1() {
    local desc="$1"; shift
    echo "--- [$desc]"
    env "$@" TRSM_PROFILE=2 numactl -N $node $MEMOPT ./trsm_test_c3 17024 512 1 2>&1 | grep -E "PASS|FAIL|prof\]|rror|fault|illed" | tail -3
    [[ -n "$hbm" ]] && grep -E "MemFree" /sys/devices/system/node/node$hbm/meminfo
}
run1 "case3 pp=1 默认(pp→HBM preferred, Xp→DDR)" A=1
run1 "case3 pp=1 pp→DDR" TRSM_LNODE0=1
run1 "case3 pp=1 pp→DDR xp→HBM" TRSM_LNODE0=1 TRSM_XPNODE=-1
run1 "case3 pp=1 LA=1" TRSM_PACK_LA=1
run1 "case3 pp=1 LA=4" TRSM_PACK_LA=4
run1 "case3 pp=1 kc=256" TRSM_KC=256
run1 "case3 pp=1 mb192 kc96" TRSM_MB=192 TRSM_KC=96
run1 "case3 pp=1 mb192 kc96 pp→DDR" TRSM_MB=192 TRSM_KC=96 TRSM_LNODE0=1
run1 "case3 pp=1 splitf=6" TRSM_SPLITF=6
run1 "case3 pp=1 nosplit" TRSM_SPLITF=0
echo "=== DONE"; date
