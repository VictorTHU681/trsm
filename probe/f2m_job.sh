#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
rm -f trsm_test; gcc $FLAGS bench_trsm.c trsm.c -o trsm_test -lm -lkblas 2>&1 | grep -E "rror" | head -5
for v in "TRSM_OWNER_PERM=0 TRSM_OWNER_HELP=0" "TRSM_OWNER_PERM=1 TRSM_OWNER_HELP=0" "TRSM_OWNER_PERM=1 TRSM_OWNER_HELP=1"; do echo "[$v]"; env $v TRSM_PROFILE=3 OMP_NUM_THREADS=38 timeout 120 numactl -N 1 ./trsm_test 512 19968 1 2>&1 | grep -E "WD|尾部不均衡|尾部协助|每线程收尾|PASS|FAIL" | tail -4 | cut -c1-300 | sed 's/.*task 合计/task 合计/'; done
echo "--- bench 5 rounds interleaved"
for i in 1 2 3 4 5; do for v in "TRSM_OWNER_PERM=0 TRSM_OWNER_HELP=0" "TRSM_OWNER_PERM=1 TRSM_OWNER_HELP=0" "TRSM_OWNER_PERM=1 TRSM_OWNER_HELP=1" "TRSM_OWNER_PERM=0 TRSM_OWNER_HELP=1"; do printf "[%s] " "$v"; env $v OMP_NUM_THREADS=38 timeout 120 numactl -N 1 ./trsm_test 512 19968 1 2>&1 | grep -E "WD|PASS|FAIL"; done; done
echo "--- correctness"
gcc -O3 -fopenmp -DTRSM_NO_SME_ASM -Dl_trsm=l_trsm_ref -c trsm.c -o probe/ref.o 2>&1 | grep -E "error"
gcc $FLAGS probe/gen_test.c trsm.c probe/ref.o -o probe/gen_test -lm 2>&1 | grep -E "error"
echo "[owner PERM=1 HELP=1]"; TRSM_OWNER=1 timeout 600 numactl -N 1 ./probe/gen_test 40 2>&1 | grep -E "FAIL|WD|PASS" | head -6
echo "[default]"; timeout 600 numactl -N 1 ./probe/gen_test 30 2>&1 | grep -E "FAIL|WD|PASS" | head -3
echo "=== DONE $(date +%H:%M:%S)"
