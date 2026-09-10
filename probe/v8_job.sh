#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST"; hostname; date
set +u; . /work_ssd/software/HPCKit/26.1.RC1/setvars.sh --use-gcc >/dev/null 2>&1; set -u
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
gcc $FLAGS bench_trsm.c trsm.c -o trsm_test -lm -lkblas 2>&1 | grep -E "error"; [[ -x trsm_test ]] || { echo COMPILE_FAIL; exit 1; }
gcc -O3 -fopenmp -DTRSM_NO_SME_ASM -Dl_trsm=l_trsm_ref -c trsm.c -o probe/ref.o 2>&1 | grep -E "error"
gcc $FLAGS probe/gen_test.c trsm.c probe/ref.o -o probe/gen_test -lm 2>&1 | grep -E "error"; [[ -x probe/gen_test ]] || { echo GEN_COMPILE_FAIL; exit 1; }
node=1; hbm=17
export OMP_NUM_THREADS=38 OMP_PROC_BIND=close OMP_PLACES=cores OMP_DYNAMIC=FALSE OMP_WAIT_POLICY=active GOMP_SPINCOUNT=infinite
echo "=== 泛化正确性"
timeout 600 numactl -N $node ./probe/gen_test 30 2>&1 | tail -1
TRSM_JIT=1 timeout 600 numactl -N $node ./probe/gen_test 30 2>&1 | tail -1
TRSM_JIT=1 TRSM_MB=192 TRSM_KC=96 timeout 600 numactl -N $node ./probe/gen_test 20 2>&1 | tail -1
TRSM_MB=192 TRSM_KC=96 TRSM_SPLITF=40 timeout 600 numactl -N $node ./probe/gen_test 20 2>&1 | tail -1
TRSM_JIT=1 TRSM_MB=64 TRSM_NC=128 TRSM_KC=32 timeout 600 numactl -N $node ./probe/gen_test 20 2>&1 | tail -1
TRSM_JIT=1 TRSM_PACK=0 TRSM_NC=96 timeout 600 numactl -N $node ./probe/gen_test 20 2>&1 | tail -1
run3() {
    local desc="$1"; shift
    echo "--- [$desc]"
    for args in "512 19968 1" "2432 17024 1" "17024 512 1"; do
        env "$@" TRSM_PROFILE=2 numactl -N $node --preferred=$hbm ./trsm_test $args 2>&1 | grep -E "PASS|FAIL|prof\]  |Error" | tail -2
    done
}
run3 "mb256 kc128" A=1
run3 "mb192 kc96 splitf12 xp=hbm" TRSM_MB=192 TRSM_KC=96 TRSM_SPLITF=12 TRSM_MAXSEG=8 TRSM_XPNODE=-1
run3 "mb192 kc96 splitf12 xp=ddr" TRSM_MB=192 TRSM_KC=96 TRSM_SPLITF=12 TRSM_MAXSEG=8 TRSM_XPNODE=-2
run3 "mb256 kc128 splitf12 xp=hbm" TRSM_SPLITF=12 TRSM_MAXSEG=8 TRSM_XPNODE=-1
run3 "mb192 kc96 pf=0" TRSM_MB=192 TRSM_KC=96 TRSM_PF=0
run3 "mb192 kc96 nt=0" TRSM_MB=192 TRSM_KC=96 TRSM_NT=0
echo "=== DONE"; date
