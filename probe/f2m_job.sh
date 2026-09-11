#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
rm -f trsm_test; gcc $FLAGS bench_trsm.c trsm.c -o trsm_test -lm -lkblas 2>&1 | grep -E "rror" | head -5
for i in 1 2; do TRSM_OWNER_HELP=1 TRSM_PROFILE=3 OMP_NUM_THREADS=38 timeout 120 numactl -N 1 ./trsm_test 512 19968 1 2>&1 | grep -E "WD|每线程平均|尾部协助|PASS|FAIL" | tail -3 | cut -c1-420; done
for i in 1 2 3 4 5; do for v in 0 1; do printf "[HELP=%s] " $v; TRSM_OWNER_HELP=$v OMP_NUM_THREADS=38 timeout 120 numactl -N 1 ./trsm_test 512 19968 1 2>&1 | grep -E "WD|PASS|FAIL"; done; done
echo "--- correctness: gen_test forced owner"
gcc -O3 -fopenmp -DTRSM_NO_SME_ASM -Dl_trsm=l_trsm_ref -c trsm.c -o probe/ref.o 2>&1 | grep -E "error"
gcc $FLAGS probe/gen_test.c trsm.c probe/ref.o -o probe/gen_test -lm 2>&1 | grep -E "error"
echo "[owner HELP=0]"; TRSM_OWNER=1 TRSM_OWNER_HELP=0 timeout 600 numactl -N 1 ./probe/gen_test 40 2>&1 | grep -E "FAIL|WD" | head -6
echo "[owner HELP=1]"; TRSM_OWNER=1 TRSM_OWNER_HELP=1 timeout 600 numactl -N 1 ./probe/gen_test 40 2>&1 | grep -E "FAIL|WD" | head -6
echo "[default]"; timeout 600 numactl -N 1 ./probe/gen_test 30 2>&1 | grep -E "FAIL|WD|PASS" | head -3
echo "=== DONE $(date +%H:%M:%S)"
