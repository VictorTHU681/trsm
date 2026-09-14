#!/usr/bin/env bash
# 定稿验证：run.sh（判题方式 + HBM preferred）两轮；纯 `numactl -N 1`（无 --preferred，题面原样）两轮；泛化正确性
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST"; hostname; date
echo "=== run.sh 第 1 轮"; bash ./run.sh 2>&1 | grep -E "PASS|FAIL|回退|rror"
echo "=== run.sh 第 2 轮"; bash ./run.sh 2>&1 | grep -E "PASS|FAIL"
. probe/env.sh
echo "=== 题面原样：OMP_NUM_THREADS=38 numactl -N 1 ./trsm_test（无 --preferred）"
for r in 1 2; do
  for args in "512 19968 1" "2432 17024 1" "17024 512 1"; do
    OMP_NUM_THREADS=38 numactl -N 1 ./trsm_test $args 2>&1 | grep -E "PASS|FAIL"
  done
done
echo "=== 泛化正确性（含 LTO 档）"
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
gcc -O3 -fopenmp -DTRSM_NO_SME_ASM -Dl_trsm=l_trsm_ref -c trsm.c -o probe/ref.o 2>&1 | grep -E "error"
gcc $FLAGS probe/gen_test.c trsm.c probe/ref.o -o probe/gen_test -lm 2>&1 | grep -E "error"
timeout 900 numactl -N 1 ./probe/gen_test 80 2>&1 | tail -1
TRSM_CFOLD=1 TRSM_OWNER=1 timeout 600 numactl -N 1 ./probe/gen_test 30 2>&1 | tail -1
TRSM_DXW=1 TRSM_OWNER=1 timeout 600 numactl -N 1 ./probe/gen_test 20 2>&1 | tail -1
TRSM_STEAL=1 TRSM_OWNER=1 OMP_NUM_THREADS=7 timeout 600 numactl -N 1 ./probe/gen_test 20 2>&1 | tail -1
TRSM_XONE=0 timeout 600 numactl -N 1 ./probe/gen_test 25 2>&1 | tail -1
TRSM_SWHOST=1 TRSM_MB=64 timeout 600 numactl -N 1 ./probe/gen_test 40 2>&1 | tail -1
TRSM_JIT=1 timeout 600 numactl -N 1 ./probe/gen_test 20 2>&1 | tail -1
gcc $FLAGS -mtune=tsv110 -flto probe/gen_test.c trsm.c probe/ref.o -o probe/gen_test_lto -lm 2>&1 | grep -E "error"
timeout 600 numactl -N 1 ./probe/gen_test_lto 40 2>&1 | tail -1
echo "=== DONE"; date
