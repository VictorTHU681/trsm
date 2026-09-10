#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST"; hostname; date
set +u; . /work_ssd/software/HPCKit/26.1.RC1/setvars.sh --use-gcc >/dev/null 2>&1; set -u
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
gcc $FLAGS -DTR_LOOKAHEAD=0 -DTRSM_PF_LOAD=0 probe/kbench.c -o probe/kbench_p0 -lm 2>&1 | grep -E "error"; [[ -x probe/kbench_p0 ]] || { echo KB_COMPILE_FAIL; exit 1; }
for cfg in "128 128 256" "256 128 256"; do
  echo "--- kc nc mb = $cfg (DDR)"; OMP_NUM_THREADS=1 taskset -c 40 ./probe/kbench_p0 $cfg 2>&1 | grep -E "^10c|SIG"
done
echo "=== DONE"; date
