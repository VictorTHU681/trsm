#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S) $(git log --oneline -1)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
rm -f trsm_test; gcc $FLAGS bench_trsm.c trsm.c -o trsm_test -lm -lkblas 2>&1 | grep -E "rror" | head -5
for v in 0 1; do
  echo "--- POOL=$v census (TC3, 12 timed runs)"
  TRSM_POOL=$v OMP_NUM_THREADS=38 numactl -N 1 ./trsm_test 17024 512 12 > /tmp/run_$v.txt 2>&1 &
  pid=$!
  for t in 1 2 3 4; do sleep 1.5; echo "t=$t threads=$(ls /proc/$pid/task 2>/dev/null | wc -l)"; for tk in /proc/$pid/task/*; do awk -v tid=$(basename $tk) '{printf "%s st=%s cpu=%s ut=%s ", tid, $3, $39, $14}' $tk/stat 2>/dev/null; echo; done | awk '{u=$4; sub("ut=","",u); if (u+0 > 5) print}' | sort -t= -k3 -n | head -80 | awk '{printf "%s ", $0} END{print ""}'; done
  wait $pid; grep -E "PASS|FAIL" /tmp/run_$v.txt
done
echo "=== DONE $(date +%H:%M:%S)"
