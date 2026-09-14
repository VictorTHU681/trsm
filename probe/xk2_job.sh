#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
. probe/env.sh
gcc -O2 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp probe/xk2test.c -o probe/xk2test -lm 2>&1 | grep -E "rror" | head -5
[[ -x probe/xk2test ]] || { echo BUILD FAILED; exit 1; }
numactl -N 1 taskset -c 40 ./probe/xk2test
rm -f probe/xk2test
