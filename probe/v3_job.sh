#!/usr/bin/env bash
# v3 作业：kbench（打包变体）+ 泛化正确性 + 三用例 profile（打包方式、kc、X 节点扫描）
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST"; hostname; date
set +u; . /work_ssd/software/HPCKit/26.1.RC1/setvars.sh --use-gcc >/dev/null 2>&1; set -u
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
gcc $FLAGS probe/kbench.c -o probe/kbench -lm 2>&1 | grep -E "error"; [[ -x probe/kbench ]] || { echo KB_COMPILE_FAIL; exit 1; }
gcc $FLAGS bench_trsm.c trsm.c -o trsm_test -lm -lkblas 2>&1 | grep -E "error"; [[ -x trsm_test ]] || { echo COMPILE_FAIL; exit 1; }
gcc -O3 -fopenmp -DTRSM_NO_SME_ASM -Dl_trsm=l_trsm_ref -c trsm.c -o probe/ref.o 2>&1 | grep -E "error"
gcc $FLAGS probe/gen_test.c trsm.c probe/ref.o -o probe/gen_test -lm 2>&1 | grep -E "error"; [[ -x probe/gen_test ]] || { echo GEN_COMPILE_FAIL; exit 1; }
node=1; hbm=17
echo "=== kbench（单线程 CPU 40，DDR）"
OMP_NUM_THREADS=1 taskset -c 40 ./probe/kbench 128 128 256
echo "--- kc=256"
OMP_NUM_THREADS=1 taskset -c 40 ./probe/kbench 256 128 256

export OMP_NUM_THREADS=38 OMP_PROC_BIND=close OMP_PLACES=cores OMP_DYNAMIC=FALSE OMP_WAIT_POLICY=active GOMP_SPINCOUNT=infinite
echo "=== 泛化正确性（默认 / 各打包方式 / 分裂）"
timeout 600 numactl -N $node ./probe/gen_test 30 2>&1 | tail -2
TRSM_JIT=1 TRSM_PACK=1 timeout 600 numactl -N $node ./probe/gen_test 20 2>&1 | tail -2
TRSM_JIT=1 TRSM_PACK=2 timeout 600 numactl -N $node ./probe/gen_test 20 2>&1 | tail -2
TRSM_JIT=1 TRSM_PACK=3 TRSM_MB=64 TRSM_NC=64 TRSM_KC=32 timeout 600 numactl -N $node ./probe/gen_test 20 2>&1 | tail -2
TRSM_JIT=1 TRSM_PACK=0 TRSM_MB=96 TRSM_NC=96 TRSM_KC=32 timeout 600 numactl -N $node ./probe/gen_test 20 2>&1 | tail -2
TRSM_SPLITF=40 TRSM_MAXSEG=4 TRSM_MB=64 TRSM_KC=32 timeout 600 numactl -N $node ./probe/gen_test 20 2>&1 | tail -2
TRSM_JIT=1 TRSM_SPLITF=40 TRSM_MB=64 TRSM_KC=32 timeout 600 numactl -N $node ./probe/gen_test 20 2>&1 | tail -2
run3() {
    local desc="$1"; shift
    echo "--- [$desc]"
    for args in "512 19968 1" "2432 17024 1" "17024 512 1"; do
        env "$@" TRSM_PROFILE=2 numactl -N $node --preferred=$hbm ./trsm_test $args 2>&1 | grep -E "PASS|FAIL|prof\]|Error" | tail -3
    done
}
echo "=== 三用例分相 profile"
run3 "default: pack=ld4 kc128 split6" A=1
run3 "kc256" TRSM_KC=256
run3 "kc256 pack=ld2" TRSM_KC=256 TRSM_PACK=2
run3 "kc256 xnode=ddr" TRSM_KC=256 TRSM_XNODE=-2
run3 "kc256 nosplit" TRSM_KC=256 TRSM_SPLITF=0
run3 "kc256 splitf=12 maxseg=8" TRSM_KC=256 TRSM_SPLITF=12 TRSM_MAXSEG=8
run3 "kc128 mb192 kc192->192" TRSM_MB=192 TRSM_KC=192
echo "=== DONE"; date
