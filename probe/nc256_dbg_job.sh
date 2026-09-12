#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(git log --oneline -1)"
. probe/env.sh
FLAGS="-O2 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
rm -f probe/nc256_dbg
gcc $FLAGS probe/nc256_dbg.c trsm.c -o probe/nc256_dbg -lm 2>&1 | grep -E "rror" | head -5
[[ -x probe/nc256_dbg ]] || { echo COMPILE_FAIL; exit 1; }
for env in "TRSM_NC=128" "TRSM_NC=256 TRSM_ABHOIST=0"; do
  for mn in "1536 512" "1536 256" "17024 512"; do
    echo "--- [$env] m n = $mn"
    env $env TRSM_OWNER=0 OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./probe/nc256_dbg $mn 2>&1 | head -8
  done
done
echo "=== DONE"
