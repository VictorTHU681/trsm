#!/usr/bin/env bash
# 多节点并发读带宽：本地 HBM17、远端 HBM 16/18/19、DDR 0/1/2/3 及组合
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST"; hostname; date
. probe/env.sh
echo "=== distance: $(cat /sys/devices/system/node/node1/distance)"
for nn in /sys/devices/system/node/node*; do n=${nn##*node}; f=$(awk '/MemFree/{print int($4/1024)}' $nn/meminfo); c=$(cat $nn/cpulist); echo "node$n free=${f}MB cpus=$c"; done | tr '\n' ';'; echo
gcc -O2 -march=armv9-a+sve2 -fopenmp probe/bw_nodes.c -o probe/bw_nodes || exit 1
run() { numactl -N 1 ./probe/bw_nodes 32 "$1"; }
for s in 17 1 0 2 3 16 18 19 20 "17,1" "17,0" "17,1,0" "17,1,0,2" "17,1,0,2,3" "17,16" "17,16,18" "17,16,18,19" "17,16,18,19,1,0,2,3" "17,17,17,1" "17,17,1,0"; do run "$s"; done
echo "=== DONE"; date
