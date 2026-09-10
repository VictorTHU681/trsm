#!/usr/bin/env bash
# 单线程内核微基准 + 三用例分相 profile
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST"; hostname; date
set +u; . /work_ssd/software/HPCKit/26.1.RC1/setvars.sh --use-gcc >/dev/null 2>&1; set -u
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
gcc $FLAGS probe/kbench.c -o probe/kbench -lm 2>&1 | grep -E "error"; [[ -x probe/kbench ]] || { echo KB_COMPILE_FAIL; exit 1; }
gcc $FLAGS bench_trsm.c trsm.c -o trsm_test -lm -lkblas 2>&1 | grep -E "error"; [[ -x trsm_test ]] || { echo COMPILE_FAIL; exit 1; }
node=1; hbm=17
echo "=== kbench（单线程，CPU 40，数据默认策略=DDR）"
OMP_NUM_THREADS=1 taskset -c 40 ./probe/kbench 128 128 256
echo "--- kc=256"
OMP_NUM_THREADS=1 taskset -c 40 ./probe/kbench 256 128 256
echo "--- 大数组放 HBM"
OMP_NUM_THREADS=1 numactl -m $hbm taskset -c 40 ./probe/kbench 128 128 256

export OMP_NUM_THREADS=38 OMP_PROC_BIND=close OMP_PLACES=cores OMP_DYNAMIC=FALSE OMP_WAIT_POLICY=active GOMP_SPINCOUNT=infinite
echo "=== 三用例分相 profile（默认参数）"
for args in "512 19968 1" "2432 17024 1" "17024 512 1"; do
    TRSM_PROFILE=2 numactl -N $node --preferred=$hbm ./trsm_test $args 2>&1 | grep -E "PASS|FAIL|prof\]" | tail -3
done
echo "=== 关闭软件预取（TRSM_PF=0）"
for args in "512 19968 1" "2432 17024 1" "17024 512 1"; do
    TRSM_PF=0 TRSM_PROFILE=2 numactl -N $node --preferred=$hbm ./trsm_test $args 2>&1 | grep -E "PASS|FAIL|prof\]" | tail -3
done
echo "=== DONE"; date
