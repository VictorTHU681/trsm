#!/usr/bin/env bash
# TC1 线程数扫描：kloop cyc/step 随并发的变化（争用定性）
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S) $(git log --oneline -1)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
rm -f trsm_test; gcc $FLAGS bench_trsm.c trsm.c -o trsm_test -lm -lkblas 2>&1 | grep -E "rror" | head -5
for nth in 38 28 19 10; do
  echo "--- nth=$nth"
  TRSM_PROFILE=2 OMP_NUM_THREADS=$nth numactl -N 1 $MEMOPT ./trsm_test 512 19968 1 2>&1 | grep -E "PASS|FAIL|每线程平均" | tail -3
done
echo "=== DONE $(date +%H:%M:%S)"
