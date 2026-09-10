#!/usr/bin/env bash
# 共享读带宽：多核同时读同一缓冲（L2 间转发？）vs 各读各的
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST"; hostname; date
. probe/env.sh
gcc -O2 -march=armv9-a+sve2 -fopenmp probe/bw_nodes.c -o probe/bw_nodes || exit 1
run() { numactl -N 1 ./probe/bw_nodes "$@"; }
echo "--- 32 MB/线程，HBM17"
run 32 17 1 0
run 32 17 2 0
run 32 17 4 0
run 32 17 19 0
run 32 17 38 0
run 32 17 38 64
run 32 17 38 1024
run 32 17 4 64
echo "--- 8 MB，HBM17（38 核聚合 L2 29 MB）"
run 8 17 38 0
run 8 17 38 64
run 8 17 4 0
echo "--- DDR1"
run 32 1 1 0
run 32 1 38 0
run 32 1 4 0
echo "=== DONE"; date
