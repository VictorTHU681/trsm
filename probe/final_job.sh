#!/usr/bin/env bash
# 判题方式：直接执行 run.sh（编译 + 三用例），外加泛化正确性
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST"; hostname; date
bash ./run.sh 2>&1 | grep -vE "^$"
echo "=== 泛化正确性"
set +u; . /work_ssd/software/HPCKit/26.1.RC1/setvars.sh --use-gcc >/dev/null 2>&1; set -u
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
gcc -O3 -fopenmp -DTRSM_NO_SME_ASM -Dl_trsm=l_trsm_ref -c trsm.c -o probe/ref.o 2>&1 | grep -E "error"
gcc $FLAGS probe/gen_test.c trsm.c probe/ref.o -o probe/gen_test -lm 2>&1 | grep -E "error"
export OMP_NUM_THREADS=38 OMP_PROC_BIND=close OMP_PLACES=cores
timeout 900 numactl -N 1 ./probe/gen_test 60 2>&1 | tail -1
TRSM_JIT=1 timeout 900 numactl -N 1 ./probe/gen_test 40 2>&1 | tail -1
echo "=== 再跑一遍 run.sh 看波动"
bash ./run.sh 2>&1 | grep -E "PASS|FAIL"
echo "=== DONE"; date
