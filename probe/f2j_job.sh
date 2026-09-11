#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S)"
. probe/env.sh
gcc -O2 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp probe/kdup2.c -o probe/kdup2 -lm 2>&1 | grep -E "error" | head
for i in 1 2; do numactl -N 1 -m 17 taskset -c 40 ./probe/kdup2; done
echo "--- 4 threads concurrently"
for c in 40 44 48 52; do numactl -N 1 -m 17 taskset -c $c ./probe/kdup2 > /tmp/kdup2_$c.txt & done; wait; cat /tmp/kdup2_40.txt
echo "=== DONE $(date +%H:%M:%S)"
