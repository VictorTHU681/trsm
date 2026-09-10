#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST"; hostname; date
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
gcc $FLAGS bench_trsm.c trsm.c -o trsm_test_c2 -lm -lkblas 2>&1 | grep -E "error"; [[ -x trsm_test_c2 ]] || { echo COMPILE_FAIL; exit 1; }
run2() {
    local desc="$1"; shift
    echo "--- [$desc]"
    env "$@" TRSM_PROFILE=2 numactl -N $node $MEMOPT ./trsm_test_c2 2432 17024 1 2>&1 | grep -E "PASS|FAIL|prof\]  |Error" | tail -2
}
run2 "mb256 kc128 (默认)" A=1
run2 "mb192 kc96" TRSM_MB=192 TRSM_KC=96
run2 "mb128 kc128" TRSM_MB=128 TRSM_KC=128
run2 "mb192 kc96 lsplit=0" TRSM_MB=192 TRSM_KC=96 TRSM_LSPLIT=0
run2 "mb192 kc96 xp=ddr" TRSM_MB=192 TRSM_KC=96 TRSM_XPNODE=-2
run2 "mb256 nc96 kc128" TRSM_NC=96
run2 "mb192 nc96 kc96" TRSM_MB=192 TRSM_NC=96 TRSM_KC=96
run2 "mb192 kc96 pf=0" TRSM_MB=192 TRSM_KC=96 TRSM_PF=0
run2 "mb192 kc192" TRSM_MB=192 TRSM_KC=192
run2 "mb160 kc160" TRSM_MB=160 TRSM_KC=160
echo "=== DONE"; date
