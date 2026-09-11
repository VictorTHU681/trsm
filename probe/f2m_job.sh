#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S)"
. probe/env.sh
uptime; ps -eo pid,pcpu,comm --sort=-pcpu | head -5
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
rm -f trsm_test; gcc $FLAGS bench_trsm.c trsm.c -o trsm_test -lm -lkblas 2>&1 | grep -E "rror" | head -5
for i in 1 2 3 4 5; do for c0 in 192 128 96 64; do printf "[C0=%s] " $c0; TRSM_OWNER_C0=$c0 OMP_NUM_THREADS=38 numactl -N 1 ./trsm_test 512 19968 1 2>&1 | grep -E "PASS|FAIL"; done; done
for c0 in 192 128 64; do echo "[C0=$c0]"; TRSM_OWNER_C0=$c0 TRSM_PROFILE=3 OMP_NUM_THREADS=38 numactl -N 1 ./trsm_test 512 19968 1 2>&1 | grep -E "尾部|每线程收尾|每线程任务" | tail -3 | sed 's/.*task 合计/task 合计/'; done
echo "=== DONE $(date +%H:%M:%S)"
