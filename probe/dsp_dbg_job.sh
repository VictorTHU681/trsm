#!/usr/bin/env bash
# DSPLIT 段错误二分：TC2 单用例，配置 = 关 / 只建包(DSPX=0) / 全开
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S)"
. probe/env.sh
FLAGS="-O3 -g -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
rm -f fastbench; gcc $FLAGS probe/fastbench.c trsm.c -o fastbench -lm 2>&1 | grep -E "rror" | head -5
[[ -x fastbench ]] || { echo COMPILE_FAIL; exit 1; }
ulimit -c unlimited
command -v gdb && echo has-gdb || echo no-gdb
for v in "TRSM_DSPLIT=0" "TRSM_DSPLIT=1 TRSM_DSPX=0" "TRSM_DSPLIT=1"; do
    echo "--- [$v] TC2"
    rm -f core core.*
    env $v OMP_NUM_THREADS=38 stdbuf -o0 -e0 numactl -N 1 $MEMOPT ./fastbench -r 1 2432 17024 2>&1 | tail -2
    echo "exit=$?"
    co=$(ls -t core* 2>/dev/null | head -1)
    if [[ -n "${co:-}" ]] && command -v gdb >/dev/null 2>&1; then
        gdb -batch -ex "bt 15" ./fastbench "$co" 2>/dev/null | grep -E "^#" | head -16
    fi
done
echo "--- [TRSM_DSPLIT=1] TC3"
rm -f core core.*
TRSM_DSPLIT=1 OMP_NUM_THREADS=38 stdbuf -o0 -e0 numactl -N 1 $MEMOPT ./fastbench -r 1 17024 512 2>&1 | tail -2
echo "exit=$?"
co=$(ls -t core* 2>/dev/null | head -1)
[[ -n "${co:-}" ]] && command -v gdb >/dev/null 2>&1 && gdb -batch -ex "bt 15" ./fastbench "$co" 2>/dev/null | grep -E "^#" | head -16
echo "=== DONE $(date +%H:%M:%S)"
