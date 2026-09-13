#!/usr/bin/env bash
echo "=== HOST $(hostname)"
for i in $(seq 0 15); do
    [[ -r /sys/devices/system/node/node$i/distance ]] || continue
    read -ra d < /sys/devices/system/node/node$i/distance
    best=999; bn=-1
    for n in "${!d[@]}"; do
        cl=$(cat /sys/devices/system/node/node$n/cpulist 2>/dev/null)
        [[ -z "$cl" ]] || continue   # 只看纯内存节点
        if (( d[n] < best )); then best=${d[n]}; bn=$n; fi
    done
    printf "node%-2d 最近HBM=node%d 距离=%d  --preferred测试:" "$i" "$bn" "$best"
    numactl -N $i --preferred=$bn true 2>&1 && echo OK || echo FAIL
done
