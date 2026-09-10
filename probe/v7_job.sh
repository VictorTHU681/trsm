#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST"; hostname; date
set +u; . /work_ssd/software/HPCKit/26.1.RC1/setvars.sh --use-gcc >/dev/null 2>&1; set -u
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
gcc $FLAGS bench_trsm.c trsm.c -o trsm_test -lm -lkblas 2>&1 | grep -E "error"; [[ -x trsm_test ]] || { echo COMPILE_FAIL; exit 1; }
node=1; hbm=17
export OMP_NUM_THREADS=38 OMP_PROC_BIND=close OMP_PLACES=cores OMP_DYNAMIC=FALSE OMP_WAIT_POLICY=active GOMP_SPINCOUNT=infinite
run3() {
    local desc="$1"; shift
    echo "--- [$desc]"
    for args in "512 19968 1" "2432 17024 1" "17024 512 1"; do
        env "$@" TRSM_PROFILE=2 numactl -N $node --preferred=$hbm ./trsm_test $args 2>&1 | grep -E "PASS|FAIL|prof\]  |Error" | tail -2
    done
}
run3 "mb256 kc128 (默认)" A=1
run3 "mb192 kc96" TRSM_MB=192 TRSM_KC=96
run3 "mb192 kc96 splitf=12" TRSM_MB=192 TRSM_KC=96 TRSM_SPLITF=12 TRSM_MAXSEG=8
run3 "mb256 kc128 splitf=12" TRSM_SPLITF=12 TRSM_MAXSEG=8
run3 "mb192 kc96 xp=hbm" TRSM_MB=192 TRSM_KC=96 TRSM_XPNODE=-1
run3 "mb224 kc224?" TRSM_MB=224 TRSM_KC=224
run3 "mb160 kc160" TRSM_MB=160 TRSM_KC=160
run3 "mb192 kc192" TRSM_MB=192 TRSM_KC=192
echo "=== DONE"; date
