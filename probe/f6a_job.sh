#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S) $(git log --oneline -1)"
. probe/env.sh
gcc -O2 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp probe/kdup2.c -o probe/kdup2 -lm 2>&1 | grep -E "error" | head -5
test -x probe/kdup2 || { echo COMPILE_FAIL; exit 1; }
for i in 1 2; do numactl -N 1 -m 17 taskset -c 40 ./probe/kdup2 | tail -8; echo ---; done
echo "=== DONE $(date +%H:%M:%S)"
