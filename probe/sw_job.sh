#!/usr/bin/env bash
# 大 cbuf / 小 kc 组合扫描（case2 + case3），验证 L2 足迹理论：cbuf + 2×(A段+B段) ≤ L2 时 kloop 效率应上升
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST"; hostname; date
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
gcc $FLAGS bench_trsm.c trsm.c -o trsm_test_sw -lm -lkblas 2>&1 | grep -E "error"; [[ -x trsm_test_sw ]] || { echo COMPILE_FAIL; exit 1; }
run1() {
    local desc="$1"; shift
    echo "--- [$desc]"
    for args in "2432 17024 1" "17024 512 1"; do
        env "$@" TRSM_PROFILE=2 numactl -N $node $MEMOPT ./trsm_test_sw $args 2>&1 | grep -E "PASS|FAIL|prof\]  |rror|fault|illed" | tail -2
    done
}
run1 "mb192 nc128 kc96 (基线)" TRSM_MB=192 TRSM_NC=128 TRSM_KC=96
run1 "mb192 nc128 kc64"        TRSM_MB=192 TRSM_NC=128 TRSM_KC=64
run1 "mb256 nc128 kc64"        TRSM_MB=256 TRSM_NC=128 TRSM_KC=64
run1 "mb192 nc192 kc96"        TRSM_MB=192 TRSM_NC=192 TRSM_KC=96
run1 "mb192 nc192 kc64"        TRSM_MB=192 TRSM_NC=192 TRSM_KC=64
run1 "mb288 nc192 kc96"        TRSM_MB=288 TRSM_NC=192 TRSM_KC=96
run1 "mb256 nc192 kc64"        TRSM_MB=256 TRSM_NC=192 TRSM_KC=64
run1 "mb288 nc160 kc96"        TRSM_MB=288 TRSM_NC=160 TRSM_KC=96
run1 "mb320 nc160 kc64"        TRSM_MB=320 TRSM_NC=160 TRSM_KC=64
run1 "mb384 nc128 kc64"        TRSM_MB=384 TRSM_NC=128 TRSM_KC=64
run1 "mb384 nc128 kc96"        TRSM_MB=384 TRSM_NC=128 TRSM_KC=96
run1 "mb256 nc256 kc64"        TRSM_MB=256 TRSM_NC=256 TRSM_KC=64
run1 "mb192 nc256 kc64"        TRSM_MB=192 TRSM_NC=256 TRSM_KC=64
run1 "mb192 nc256 kc96"        TRSM_MB=192 TRSM_NC=256 TRSM_KC=96
run1 "mb192 nc192 kc96 pf=0"   TRSM_MB=192 TRSM_NC=192 TRSM_KC=96 TRSM_PF=0
echo "=== DONE"; date
