#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S) $(git log --oneline -1)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
rm -f trsm_test; gcc $FLAGS bench_trsm.c trsm.c -o trsm_test -lm -lkblas 2>&1 | grep -E "rror" | head -5
for i in 1 2 3 4; do for v in 0 1; do for args in "512 19968 1" "2432 17024 1" "17024 512 1"; do printf "[NOBAR=%s] " $v; TRSM_NOBAR=$v OMP_NUM_THREADS=38 timeout 300 numactl -N 1 ./trsm_test $args 2>&1 | grep -E "PASS|FAIL"; done; done; done
for v in 0 1; do for args in "512 19968 1" "2432 17024 1"; do echo "[NOBAR=$v $args]"; TRSM_NOBAR=$v TRSM_PROFILE=2 OMP_NUM_THREADS=38 timeout 300 numactl -N 1 ./trsm_test $args 2>&1 | grep -E "每线程平均" | tail -1 | cut -c1-330; done; done
echo "--- correctness"
gcc -O3 -fopenmp -DTRSM_NO_SME_ASM -Dl_trsm=l_trsm_ref -c trsm.c -o probe/ref.o 2>&1 | grep -E "error"
gcc $FLAGS probe/gen_test.c trsm.c probe/ref.o -o probe/gen_test -lm 2>&1 | grep -E "error"
echo "[default]"; timeout 900 numactl -N 1 ./probe/gen_test 50 2>&1 | grep -E "FAIL|PASS" | head -4
echo "[owner]"; TRSM_OWNER=1 timeout 900 numactl -N 1 ./probe/gen_test 30 2>&1 | grep -E "FAIL|PASS" | head -4
echo "[5 threads]"; OMP_NUM_THREADS=5 timeout 900 numactl -N 1 ./probe/gen_test 20 2>&1 | grep -E "FAIL|PASS" | head -4
echo "[ppmode=1 forced]"; TRSM_PPMODE=1 timeout 900 numactl -N 1 ./probe/gen_test 20 2>&1 | grep -E "FAIL|PASS" | head -4
echo "[jit forced]"; TRSM_JIT_MB=0 timeout 900 numactl -N 1 ./probe/gen_test 20 2>&1 | grep -E "FAIL|PASS" | head -4
echo "=== DONE $(date +%H:%M:%S)"
