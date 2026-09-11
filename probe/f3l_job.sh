#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S) $(git log --oneline -1)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
rm -f trsm_test; gcc $FLAGS bench_trsm.c trsm.c -o trsm_test -lm -lkblas 2>&1 | grep -E "rror" | head -5
echo "--- smaps after REHUGE"
for args in "2432 17024 1" "512 19968 1"; do echo "[$args]"; TRSM_SMAPS=1 OMP_NUM_THREADS=38 numactl -N 1 ./trsm_test $args 2>&1 | grep -E "smaps.*X vma|PASS|FAIL" | tail -3; done
echo "--- correctness quick"
gcc -O3 -fopenmp -DTRSM_NO_SME_ASM -Dl_trsm=l_trsm_ref -c trsm.c -o probe/ref.o 2>&1 | grep -E "error"
gcc $FLAGS probe/gen_test.c trsm.c probe/ref.o -o probe/gen_test -lm 2>&1 | grep -E "error"
timeout 900 numactl -N 1 ./probe/gen_test 40 2>&1 | grep -E "FAIL|PASS" | head -3
echo "=== DONE $(date +%H:%M:%S)"
