#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S) $(git log --oneline -1)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
rm -f trsm_test; gcc $FLAGS bench_trsm.c trsm.c -o trsm_test -lm -lkblas 2>&1 | grep -E "rror" | head -5
for i in 1 2 3 4; do for v in 0 1; do for args in "2432 17024 1" "17024 512 1"; do printf "[XDIRECT=%s] " $v; TRSM_XDIRECT=$v OMP_NUM_THREADS=38 timeout 300 numactl -N 1 ./trsm_test $args 2>&1 | grep -E "PASS|FAIL"; done; done; done
for v in 0 1; do for args in "2432 17024 1" "17024 512 1"; do echo "[XDIRECT=$v $args]"; TRSM_XDIRECT=$v TRSM_PROFILE=2 OMP_NUM_THREADS=38 timeout 300 numactl -N 1 ./trsm_test $args 2>&1 | grep -E "每线程平均" | tail -1 | cut -c1-330; done; done
echo "--- extra shapes through special paths (XDIRECT=1)"
for args in "2432 8192 1" "1536 17024 1" "17024 256 1" "9000 512 1" "2560 4096 1" "4096 12800 1"; do printf "%-16s " "[$args]"; OMP_NUM_THREADS=38 timeout 300 numactl -N 1 ./trsm_test $args 2>&1 | grep -E "PASS|FAIL"; done
echo "--- correctness"
gcc -O3 -fopenmp -DTRSM_NO_SME_ASM -Dl_trsm=l_trsm_ref -c trsm.c -o probe/ref.o 2>&1 | grep -E "error"
gcc $FLAGS probe/gen_test.c trsm.c probe/ref.o -o probe/gen_test -lm 2>&1 | grep -E "error"
echo "[default]"; timeout 900 numactl -N 1 ./probe/gen_test 40 2>&1 | grep -E "FAIL|PASS" | head -4
echo "=== DONE $(date +%H:%M:%S)"
