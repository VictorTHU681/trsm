#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S) $(git log --oneline -1)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
rm -f trsm_test; gcc $FLAGS bench_trsm.c trsm.c -o trsm_test -lm -lkblas 2>&1 | grep -E "rror" | head -5
echo "--- clean node: CPROBE on/off (expect identical)"
for i in 1 2 3; do for v in 1 0; do printf "[CPROBE=%s] " $v; TRSM_CPROBE=$v OMP_NUM_THREADS=38 numactl -N 1 ./trsm_test 512 19968 1 2>&1 | grep -E "PASS|FAIL"; done; done
echo "--- injected noise: spinner pinned to cpu 45 (node 1)"
taskset -c 45 bash -c 'while :; do :; done' & SP=$!
sleep 1
for i in 1 2 3; do for v in 0 1; do printf "[noise CPROBE=%s] " $v; TRSM_CPROBE=$v TRSM_PROFILE=1 OMP_NUM_THREADS=38 numactl -N 1 ./trsm_test 512 19968 1 2>&1 | grep -E "cprobe|PASS|FAIL" | tr '\n' ' '; echo; done; done
for v in 0 1; do printf "[noise TC3 CPROBE=%s] " $v; TRSM_CPROBE=$v OMP_NUM_THREADS=38 numactl -N 1 ./trsm_test 17024 512 1 2>&1 | grep -E "PASS|FAIL"; done
kill $SP 2>/dev/null
echo "--- correctness"
gcc -O3 -fopenmp -DTRSM_NO_SME_ASM -Dl_trsm=l_trsm_ref -c trsm.c -o probe/ref.o 2>&1 | grep -E "error"
gcc $FLAGS probe/gen_test.c trsm.c probe/ref.o -o probe/gen_test -lm 2>&1 | grep -E "error"
timeout 900 numactl -N 1 ./probe/gen_test 40 2>&1 | grep -E "FAIL|PASS" | head -3
echo "=== DONE $(date +%H:%M:%S)"
