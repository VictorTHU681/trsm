#!/usr/bin/env bash
# gdb 下跑 DSPLIT=1 TC2，抓 SIGSEGV 回溯
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S)"
. probe/env.sh
FLAGS="-O2 -g -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
rm -f fastbench_g; gcc $FLAGS probe/fastbench.c trsm.c -o fastbench_g -lm 2>&1 | grep -E "rror" | head -5
[[ -x fastbench_g ]] || { echo COMPILE_FAIL; exit 1; }
export TRSM_DSPLIT=1 OMP_NUM_THREADS=38
numactl -N 1 $MEMOPT gdb -batch -q \
  -ex "handle SIGSEGV stop nopass" \
  -ex run \
  -ex "bt 20" \
  -ex "info registers pc x0 x1 x2 x3" \
  --args ./fastbench_g -r 1 2432 17024 2>&1 | grep -vE "^\[New Thread|^\[Thread|guile|objcode|;;;"
echo "=== DONE $(date +%H:%M:%S)"
