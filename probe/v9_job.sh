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
echo "=== 泛化正确性（含流水式打包路径：TRSM_JIT_MB=0 强制大 L 模式）"
timeout 600 numactl -N $node ./probe/gen_test 30 2>&1 | tail -1
TRSM_JIT_MB=0 timeout 600 numactl -N $node ./probe/gen_test 30 2>&1 | tail -1
TRSM_JIT_MB=0 TRSM_MB=96 TRSM_KC=32 TRSM_PACK_LA=1 timeout 600 numactl -N $node ./probe/gen_test 20 2>&1 | tail -1
TRSM_JIT_MB=0 TRSM_SPLITF=40 TRSM_SPLIT_MINCH=2 TRSM_MB=64 TRSM_KC=32 timeout 600 numactl -N $node ./probe/gen_test 20 2>&1 | tail -1
run1() {   # 只跑 case3
    local desc="$1"; shift
    echo "--- [$desc]"
    env "$@" TRSM_PROFILE=2 numactl -N $node --preferred=$hbm ./trsm_test 17024 512 1 2>&1 | grep -E "PASS|FAIL|prof\]|Error" | tail -3
}
run1 "case3 pp=1 默认(LA=2)" A=1
run1 "case3 pp=1 LA=1" TRSM_PACK_LA=1
run1 "case3 pp=1 LA=4" TRSM_PACK_LA=4
run1 "case3 pp=1 kc=256" TRSM_KC=256
run1 "case3 pp=1 mb192 kc96" TRSM_MB=192 TRSM_KC=96
run1 "case3 pp=1 xp=hbm" TRSM_XPNODE=-1
run1 "case3 pp=1 pp-node=ddr" TRSM_LNODE0=1
run1 "case3 pp=0 (JIT 对照)" TRSM_PPMODE=0
echo "=== case1/2 回归"
for args in "512 19968 1" "2432 17024 1"; do TRSM_PROFILE=2 numactl -N $node --preferred=$hbm ./trsm_test $args 2>&1 | grep -E "PASS|FAIL|prof\]  " | tail -2; done
echo "=== DONE"; date
