#!/usr/bin/env bash
# 快速验证：题面原样命令两轮（含 profile）+ 泛化 40 组
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST"; hostname; date
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
gcc $FLAGS bench_trsm.c trsm.c -o trsm_test_q -lm -lkblas 2>&1 | grep -E "error"; [[ -x trsm_test_q ]] || { echo COMPILE_FAIL; exit 1; }
for r in 1 2; do
  echo "--- 第 $r 轮（题面原样，无 --preferred）"
  for args in "512 19968 1" "2432 17024 1" "17024 512 1"; do
    OMP_NUM_THREADS=38 TRSM_PROFILE=2 numactl -N 1 ./trsm_test_q $args 2>&1 | grep -E "PASS|FAIL|prof\]  每|rror|fault|illed" | sed -E "s/每线程平均 ms：prep [0-9.]+ \| //; s/xcopy 0.000 //; s/ \| task 合计.*理想 [0-9.]+ ms，/ | /; s/pack 0.000 //; s/ +/ /g; s/wait 0.00[0-9] //"
  done
done
echo "=== 泛化正确性"
gcc -O3 -fopenmp -DTRSM_NO_SME_ASM -Dl_trsm=l_trsm_ref -c trsm.c -o probe/ref.o 2>&1 | grep -E "error"
gcc $FLAGS probe/gen_test.c trsm.c probe/ref.o -o probe/gen_test -lm 2>&1 | grep -E "error"
timeout 900 numactl -N 1 ./probe/gen_test 40 2>&1 | tail -1
echo "=== DONE"; date
