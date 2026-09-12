#!/usr/bin/env bash
# fresh18 CSHIFT 正式认证：官方 bench，判题口径（numactl -N 1 $MEMOPT），开/关交错。
# TC2：auto(on) vs off ×6 轮 ×5 计时；TC1：owner 强开(=2) vs off ×6 轮 ×5；TC3：auto(≡off) vs off ×2 轮 ×3（门控恒等性）。
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S) $(git log --oneline -1)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
rm -f trsm_test; gcc $FLAGS bench_trsm.c trsm.c -o trsm_test -lm -lkblas 2>&1 | grep -E "rror" | head -5
[[ -x trsm_test ]] || { echo COMPILE_FAIL; exit 1; }
run() { # run <tag> <env...> -- M N R
    local tag="$1"; shift
    local envs=()
    while [[ "$1" != "--" ]]; do envs+=("$1"); shift; done; shift
    env "${envs[@]}" OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./trsm_test "$1" "$2" "$3" 2>&1 | grep -E "PASS|FAIL" | sed "s/^/[$tag] /"
}
for i in 1 2 3 4 5 6; do
    echo "--- round $i $(date +%H:%M:%S)"
    run "tc2 off $i" TRSM_CSHIFT=0 -- 2432 17024 5
    run "tc2 on  $i" TRSM_CSHIFT=1 -- 2432 17024 5
    run "tc1 off $i" TRSM_CSHIFT=0 -- 512 19968 5
    run "tc1 on2 $i" TRSM_CSHIFT=2 -- 512 19968 5
done
for i in 1 2; do
    run "tc3 off $i" TRSM_CSHIFT=0 -- 17024 512 3
    run "tc3 auto $i" TRSM_CSHIFT=1 -- 17024 512 3
done
echo "--- PROFILE=3 setup 分解（fastbench）"
rm -f fastbench; gcc $FLAGS probe/fastbench.c trsm.c -o fastbench -lm 2>&1 | grep -E "rror" | head -3
TRSM_PROFILE=3 OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./fastbench -r 2 2>&1 | grep -E "setup|分解" | head -24
echo "=== DONE $(date +%H:%M:%S)"
