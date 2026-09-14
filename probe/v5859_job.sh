#!/usr/bin/env bash
# v5859：v58（b34d492）vs v59（当前树）同节点交错 A/B——验证 v59 无回归（P3 env 快照 + 惰性实验代码应中性）
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S) $(git log --oneline -1)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
git show b34d492:trsm.c > /tmp/trsm_v58_$$.c
rm -f trsm_v58b trsm_v59b
gcc $FLAGS bench_trsm.c /tmp/trsm_v58_$$.c -o trsm_v58b -lm -lkblas 2>&1 | grep -E "rror" | head -3
gcc $FLAGS bench_trsm.c trsm.c -o trsm_v59b -lm -lkblas 2>&1 | grep -E "rror" | head -3
[[ -x trsm_v58b && -x trsm_v59b ]] || { echo "BUILD FAILED"; exit 1; }
sha256sum /tmp/trsm_v58_$$.c trsm.c | cut -c1-24
run1(){ local lbl="$1" bin="$2"; shift 2; echo "--- $lbl"; env "$@" OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./$bin ${ARGS} 2>&1 | grep -E " x |PASS|FAIL"; }
echo "=== TC1 ABBA x6"
for r in 1 2 3 4 5 6; do
  ARGS="512 19968 5" run1 "tc1_v58_a$r" trsm_v58b
  ARGS="512 19968 5" run1 "tc1_v59_a$r" trsm_v59b
  ARGS="512 19968 5" run1 "tc1_v59_b$r" trsm_v59b
  ARGS="512 19968 5" run1 "tc1_v58_b$r" trsm_v58b
done
echo "=== TC2/TC3 ABBA x3"
for r in 1 2 3; do
  for args in "2432 17024 3" "17024 512 3"; do
    ARGS="$args" run1 "tc_${args%% *}_v58_$r" trsm_v58b
    ARGS="$args" run1 "tc_${args%% *}_v59_$r" trsm_v59b
  done
done
rm -f trsm_v58b trsm_v59b /tmp/trsm_v58_$$.c
echo "=== DONE $(date +%H:%M:%S)"
