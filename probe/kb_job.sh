#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST"; hostname; date
set +u; . /work_ssd/software/HPCKit/26.1.RC1/setvars.sh --use-gcc >/dev/null 2>&1; set -u
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
gcc $FLAGS probe/kbench.c -o probe/kbench -lm 2>&1 | grep -E "error"; [[ -x probe/kbench ]] || { echo KB_COMPILE_FAIL; exit 1; }
for cfg in "128 128 256" "256 128 256"; do
  echo "--- kc nc mb = $cfg (DDR)"; OMP_NUM_THREADS=1 taskset -c 40 ./probe/kbench $cfg 2>&1 | tee /tmp/kb_last.txt
  pc=$(grep -o 'pc=0x[0-9a-f]*' /tmp/kb_last.txt | head -1 | cut -d= -f2)
  if [[ -n "$pc" ]]; then echo "--- disasm around $pc"; objdump -d --start-address=$((pc-32)) --stop-address=$((pc+16)) probe/kbench | tail -14; fi
done
echo "=== DONE"; date
