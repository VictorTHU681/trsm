#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST"; hostname; date
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
rm -f probe/kbench; gcc $FLAGS probe/kbench.c -o probe/kbench -lm 2>&1 | grep -E "rror" | head -5; [[ -x probe/kbench ]] || { echo KB_COMPILE_FAIL; exit 1; }
OMP_NUM_THREADS=1 taskset -c 40 ./probe/kbench 64 128 192 12 2>&1 | grep -E "^12|SIG|\[kb\]" | head -8; OMP_NUM_THREADS=1 taskset -c 40 ./probe/kbench 128 128 192 12 2>&1 | grep -E "^12|SIG|\[kb\]" | head -8
echo "=== DONE"; date
