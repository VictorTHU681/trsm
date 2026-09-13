#!/usr/bin/env bash
echo "=== HOST $(hostname)"
numactl -H
echo "--- distances"
cat /sys/devices/system/node/node*/distance | head -32
echo "--- 各节点 cpu/内存"
for n in /sys/devices/system/node/node*/; do
    n=${n%/}; i=${n##*node}
    printf "node%-2s cpus=[%s] mem=%s\n" "$i" "$(cat $n/cpulist)" "$(grep MemTotal $n/meminfo | awk '{print $4/1048576 "GB"}')"
done
