#!/usr/bin/env bash
# v1 作业：编译、泛化正确性、三用例（判题口径 + profile）、几组参数对照
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST"; hostname; date
set +u; . /work_ssd/software/HPCKit/26.1.RC1/setvars.sh --use-gcc >/dev/null 2>&1; set -u
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
echo "=== compile"
gcc $FLAGS bench_trsm.c trsm.c -o trsm_test -lm -lkblas 2>&1 | grep -E "error|warning" ; [[ -x trsm_test ]] || { echo COMPILE_FAIL; exit 1; }
gcc -O3 -fopenmp -DTRSM_NO_SME_ASM -Dl_trsm=l_trsm_ref -c trsm.c -o probe/ref.o 2>&1 | grep -E "error"
gcc $FLAGS probe/gen_test.c trsm.c probe/ref.o -o probe/gen_test -lm 2>&1 | grep -E "error"; [[ -x probe/gen_test ]] || { echo GEN_COMPILE_FAIL; exit 1; }

node=1; hbm=17
export OMP_NUM_THREADS=38 OMP_PROC_BIND=close OMP_PLACES=cores OMP_DYNAMIC=FALSE OMP_WAIT_POLICY=active GOMP_SPINCOUNT=infinite
echo "=== 泛化正确性（38 线程，默认参数）"
timeout 600 numactl -N $node ./probe/gen_test 30 2>&1 | tail -8
echo "=== 泛化正确性（小 mb/nc/kc，多边界）"
TRSM_MB=32 TRSM_NC=32 TRSM_KC=16 timeout 600 numactl -N $node ./probe/gen_test 20 2>&1 | tail -4
TRSM_JIT=1 TRSM_MB=48 TRSM_NC=64 TRSM_KC=16 timeout 600 numactl -N $node ./probe/gen_test 20 2>&1 | tail -4

run3() {   # $1 = 描述，其余为环境变量赋值
    local desc="$1"; shift
    echo "--- [$desc]"
    for args in "512 19968 2" "2432 17024 2" "17024 512 1"; do
        env "$@" TRSM_PROFILE=1 numactl -N $node --preferred=$hbm ./trsm_test $args 2>&1 | grep -E "PASS|FAIL|prof\]|Error" | tail -3
    done
}
echo "=== 判题口径（--preferred=$hbm），默认参数 mb=256 nc=128 kc=128"
run3 "default" A=1
echo "=== X 迁到 DDR（TRSM_XNODE=-2）"
run3 "xnode=ddr" TRSM_XNODE=-2
echo "=== kc=256"
run3 "kc256" TRSM_KC=256
echo "=== mb=192 nc=128 kc=192"
run3 "mb192" TRSM_MB=192 TRSM_KC=192
echo "=== mb=128 nc=256 kc=128"
run3 "nc256" TRSM_MB=128 TRSM_NC=256 TRSM_KC=128
echo "=== 无 --preferred（全 DDR）对照"
for args in "512 19968 2" "2432 17024 2" "17024 512 1"; do
    TRSM_PROFILE=1 numactl -N $node ./trsm_test $args 2>&1 | grep -E "PASS|FAIL|prof\]" | tail -2
done
echo "=== DONE"; date
