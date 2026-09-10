#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST"; hostname; date
set +u; . /work_ssd/software/HPCKit/26.1.RC1/setvars.sh --use-gcc >/dev/null 2>&1; set -u
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
for la in 0 128 192 256 512; do
  gcc $FLAGS -DTR_LOOKAHEAD=$la -DTRSM_PF_LOAD=1 probe/kbench.c -o probe/kbench_la$la -lm 2>&1 | grep -E "error"; [[ -x probe/kbench_la$la ]] || { echo KB_COMPILE_FAIL; exit 1; }
  echo "--- LOOKAHEAD(ld)=$la kc=128"; OMP_NUM_THREADS=1 taskset -c 40 ./probe/kbench_la$la 128 128 256 2>&1 | grep -E "^7\.|^1[uv]|SIG"
done
echo "=== DONE"; date
