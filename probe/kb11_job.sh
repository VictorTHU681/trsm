#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST"; hostname; date
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
gcc $FLAGS probe/kbench.c -o probe/kbench -lm 2>&1 | grep -E "error"; [[ -x probe/kbench ]] || { echo KB_COMPILE_FAIL; exit 1; }
for cfg in "64 128 192" "128 128 192" "64 128 256"; do
  echo "--- kc nc mb = $cfg"
  OMP_NUM_THREADS=1 taskset -c 40 ./probe/kbench $cfg 2>&1 | grep -E "^11|^1\.|^2h|^2s|SIG" | head -12
done
echo "=== DONE"; date
