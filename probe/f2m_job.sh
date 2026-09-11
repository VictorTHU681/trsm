#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
rm -f trsm_test; gcc $FLAGS bench_trsm.c trsm.c -o trsm_test -lm -lkblas 2>&1 | grep -E "rror" | head -5
for v in 0 1; do echo "[HELP=$v]"; TRSM_OWNER_HELP=$v TRSM_PROFILE=3 OMP_NUM_THREADS=38 timeout 120 numactl -N 1 ./trsm_test 512 19968 1 2>&1 | grep -E "WD|每线程平均|尾部协助|每线程收尾|PASS|FAIL" | tail -5 | cut -c1-400; done
echo "--- correctness: gen_test forced owner (HELP=0 then 1)"
gcc -O3 -fopenmp -DTRSM_NO_SME_ASM -Dl_trsm=l_trsm_ref -c trsm.c -o probe/ref.o 2>&1 | grep -E "error"
gcc $FLAGS probe/gen_test.c trsm.c probe/ref.o -o probe/gen_test -lm 2>&1 | grep -E "error"
TRSM_OWNER=1 TRSM_OWNER_HELP=0 timeout 600 numactl -N 1 ./probe/gen_test 40 2>&1 | grep -E "FAIL|WD" | head
TRSM_OWNER=1 TRSM_OWNER_HELP=1 timeout 600 numactl -N 1 ./probe/gen_test 40 2>&1 | grep -E "FAIL|WD" | head
echo "=== DONE $(date +%H:%M:%S)"
