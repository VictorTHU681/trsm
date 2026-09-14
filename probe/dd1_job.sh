#!/usr/bin/env bash
# ddN：TC2 双 bank 对角探针（W 门 + 二分 + ALT + UNIT）
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S) $(git log --oneline -1)"
. probe/env.sh
rm -f probe/ddiag
gcc -O2 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp probe/ddiag.c -o probe/ddiag -lm 2>&1 | grep -E "rror" | head -8
[[ -x probe/ddiag ]] || { echo "BUILD FAILED"; exit 1; }
echo "-- UNIT"
DD_UNIT=1 timeout 120 numactl -N 1 taskset -c 40 ./probe/ddiag 2>&1 | grep UNIT
echo "-- 主测"
timeout 300 numactl -N 1 taskset -c 40 ./probe/ddiag 2>&1
echo "-- NEWMAX=4"
DD_NEWMAX=4 timeout 120 numactl -N 1 taskset -c 40 ./probe/ddiag 2>&1 | grep -A2 bitwise
echo "-- ALT=1"
DD_ALT=1 timeout 120 numactl -N 1 taskset -c 40 ./probe/ddiag 2>&1 | grep -A2 bitwise
rm -f probe/ddiag
echo "=== DONE $(date +%H:%M:%S)"
