#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
echo "=== 泛化正确性"
gcc -O3 -fopenmp -DTRSM_NO_SME_ASM -Dl_trsm=l_trsm_ref -c trsm.c -o probe/ref.o 2>&1 | grep -E "error"
gcc $FLAGS probe/gen_test.c trsm.c probe/ref.o -o probe/gen_test -lm 2>&1 | grep -E "error"
timeout 600 numactl -N 1 ./probe/gen_test 60 2>&1 | tail -1
TRSM_OWNER=1 timeout 600 numactl -N 1 ./probe/gen_test 40 2>&1 | tail -1
TRSM_OWNER=0 TRSM_DIRECT=0 TRSM_SPECIAL=0 timeout 600 numactl -N 1 ./probe/gen_test 40 2>&1 | tail -1
TRSM_OWNER=0 TRSM_MB=256 TRSM_DIRECT=0 timeout 600 numactl -N 1 ./probe/gen_test 30 2>&1 | tail -1
TRSM_OWNER=0 TRSM_MB=64 TRSM_NC=64 TRSM_KC=32 timeout 600 numactl -N 1 ./probe/gen_test 20 2>&1 | tail -1
TRSM_DDUP=0 timeout 600 numactl -N 1 ./probe/gen_test 20 2>&1 | tail -1
echo "=== 官方 bench 交错 A/B（DDUP）"
rm -f trsm_test; gcc $FLAGS bench_trsm.c trsm.c -o trsm_test -lm -lkblas 2>&1 | grep -E "rror" | head -5
for i in 1 2 3; do
  for v in "TRSM_DDUP=0" "TRSM_DDUP=1"; do
    for args in "512 19968 1" "2432 17024 1" "17024 512 1"; do printf "%-14s " "[$v]"; env $v OMP_NUM_THREADS=38 numactl -N 1 ./trsm_test $args 2>&1 | grep -E "PASS|FAIL"; done
  done
done
echo "=== profile case2 / case1"
for v in "TRSM_DDUP=0" "TRSM_DDUP=1"; do echo "[$v]"; env $v TRSM_PROFILE=2 OMP_NUM_THREADS=38 numactl -N 1 ./trsm_test 2432 17024 1 2>&1 | grep -E "每线程" | tail -1; env $v TRSM_PROFILE=2 OMP_NUM_THREADS=38 numactl -N 1 ./trsm_test 512 19968 1 2>&1 | grep -E "每线程" | tail -1; done
echo "=== DONE $(date +%H:%M:%S)"
