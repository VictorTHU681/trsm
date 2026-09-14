#!/usr/bin/env bash
# dd1：TC2 双 bank 对角探针（立项门：加权 ≤−8% GO；b4≈b0；bitwise PASS）
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S) $(git log --oneline -1)"
. probe/env.sh
rm -f probe/ddiag
gcc -O2 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp probe/ddiag.c -o probe/ddiag -lm 2>&1 | grep -E "rror" | head -8
[[ -x probe/ddiag ]] || { echo "BUILD FAILED"; exit 1; }
for r in 1 2 3; do
  echo "--- run$r"
  timeout 300 numactl -N 1 taskset -c 40 ./probe/ddiag 2>&1; for nm in 3 4 7 8 11 12; do echo "-- NEWMAX=$nm"; DD_NEWMAX=$nm timeout 120 numactl -N 1 taskset -c 40 ./probe/ddiag 2>&1 | grep bitwise; done
done
rm -f probe/ddiag
echo "=== DONE $(date +%H:%M:%S)"
