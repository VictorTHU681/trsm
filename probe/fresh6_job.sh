#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
rm -f fastbench; gcc $FLAGS probe/fastbench.c trsm.c -o fastbench -lm 2>&1 | grep -E "rror" | head -5; [[ -x fastbench ]] || { echo COMPILE_FAIL; exit 1; }
run1() { echo "--- [$*]"; env "$@" OMP_NUM_THREADS=38 numactl -N 1 ./fastbench -r 5 512 19968 2>&1 | grep -vE "^  rep"; }
for i in 1 2; do
run1 TRSM_DUPLEX=0 TRSM_OWNER_PF2=0
run1 TRSM_DUPLEX=0 TRSM_OWNER_PF2=1
run1 TRSM_DUPLEX=1 TRSM_OWNER_PF2=0
run1 TRSM_DUPLEX=1 TRSM_OWNER_PF2=1
done
rm -f trsm_test; gcc $FLAGS bench_trsm.c trsm.c -o trsm_test -lm -lkblas 2>&1 | grep -E "rror" | head -5
prof1() { echo "--- [$*]"; env "$@" TRSM_PROFILE=2 OMP_NUM_THREADS=38 numactl -N 1 ./trsm_test 512 19968 1 2>&1 | grep -E "PASS|FAIL|每线程" | tail -2; }
prof1 TRSM_DUPLEX=0 TRSM_OWNER_PF2=1
prof1 TRSM_DUPLEX=1 TRSM_OWNER_PF2=1
gcc -O3 -fopenmp -DTRSM_NO_SME_ASM -Dl_trsm=l_trsm_ref -c trsm.c -o probe/ref.o 2>&1 | grep -E "error"
gcc $FLAGS probe/gen_test.c trsm.c probe/ref.o -o probe/gen_test -lm 2>&1 | grep -E "error"
TRSM_OWNER=1 TRSM_OWNER_PF2=1 timeout 600 numactl -N 1 ./probe/gen_test 30 2>&1 | tail -1
echo "=== DONE $(date +%H:%M:%S)"
