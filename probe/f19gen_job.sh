#!/usr/bin/env bash
# fresh19 IRED 泛化验证：gen_test（默认 IRED=1 / IRED=0 / owner / 5 线程），大 m 形状触发分裂归约竞争
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S) $(git log --oneline -1)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
gcc -O3 -fopenmp -DTRSM_NO_SME_ASM -Dl_trsm=l_trsm_ref -c trsm.c -o probe/ref.o 2>&1 | grep -E "error"
gcc $FLAGS probe/gen_test.c trsm.c probe/ref.o -o probe/gen_test -lm 2>&1 | grep -E "error"
[[ -x probe/gen_test ]] || { echo COMPILE_FAIL; exit 1; }
echo "[ired on]"; timeout 900 numactl -N 1 ./probe/gen_test 80 2>&1 | grep -E "FAIL|PASS" | head -3
echo "[ired off]"; TRSM_IRED=0 timeout 900 numactl -N 1 ./probe/gen_test 40 2>&1 | grep -E "FAIL|PASS" | head -3
echo "[splitf4 更多分裂]"; TRSM_SPLITF=4 TRSM_MAXSEG=8 timeout 900 numactl -N 1 ./probe/gen_test 40 2>&1 | grep -E "FAIL|PASS" | head -3
echo "[owner]"; TRSM_OWNER=1 timeout 600 numactl -N 1 ./probe/gen_test 20 2>&1 | grep -E "FAIL|PASS" | head -3
echo "[5 threads]"; OMP_NUM_THREADS=5 timeout 600 numactl -N 1 ./probe/gen_test 20 2>&1 | grep -E "FAIL|PASS" | head -3
echo "=== DONE $(date +%H:%M:%S)"
