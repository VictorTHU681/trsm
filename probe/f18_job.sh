#!/usr/bin/env bash
# fresh18 CSHIFT 泛化验证：gen_test（默认 / CSHIFT=0 / owner / 少线程），随机 (m,n,lda,ldb) 覆盖奇面板路由
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S) $(git log --oneline -1)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
echo "--- build"
gcc -O3 -fopenmp -DTRSM_NO_SME_ASM -Dl_trsm=l_trsm_ref -c trsm.c -o probe/ref.o 2>&1 | grep -E "error"
gcc $FLAGS probe/gen_test.c trsm.c probe/ref.o -o probe/gen_test -lm 2>&1 | grep -E "error"
[[ -x probe/gen_test ]] || { echo COMPILE_FAIL; exit 1; }
echo "[default cshift auto]"; timeout 900 numactl -N 1 ./probe/gen_test 60 2>&1 | grep -E "FAIL|PASS" | head -4
echo "[cshift off]"; TRSM_CSHIFT=0 timeout 900 numactl -N 1 ./probe/gen_test 40 2>&1 | grep -E "FAIL|PASS" | head -4
echo "[owner]"; TRSM_OWNER=1 timeout 900 numactl -N 1 ./probe/gen_test 30 2>&1 | grep -E "FAIL|PASS" | head -4
echo "[owner cshift off]"; TRSM_OWNER=1 TRSM_CSHIFT=0 timeout 900 numactl -N 1 ./probe/gen_test 20 2>&1 | grep -E "FAIL|PASS" | head -4
echo "[5 threads]"; OMP_NUM_THREADS=5 timeout 900 numactl -N 1 ./probe/gen_test 20 2>&1 | grep -E "FAIL|PASS" | head -4
echo "=== DONE $(date +%H:%M:%S)"
