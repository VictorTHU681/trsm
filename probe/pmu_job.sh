#!/usr/bin/env bash
# TC1 vs TC2 PMU 对照（判题口径）：cycles/insn/前后端停顿/L1D/L2 重填，按用例整体计（含全部相位）
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S) $(git log --oneline -1)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
rm -f trsm_test; gcc $FLAGS bench_trsm.c trsm.c -o trsm_test -lm -lkblas 2>&1 | grep -E "rror" | head -5
command -v perf >/dev/null || { echo NO_PERF; exit 1; }
perf list 2>/dev/null | grep -iE "stall|l2d_cache_refill|l1d_cache_refill" | head -8
for MN in "512 19968" "2432 17024"; do
  read -r M N <<<"$MN"
  echo "--- perf $M x $N (3 轮)"
  for i in 1 2 3; do
    perf stat -e cycles,instructions,stall_backend,stall_frontend,l1d_cache_refill,l2d_cache_refill \
      -- env OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./trsm_test $M $N 1 2>&1 | grep -E "PASS|cycles|instructions|stall|refill|seconds" | head -10
  done
done
echo "=== DONE $(date +%H:%M:%S)"
