#!/usr/bin/env bash
echo "=== HOST $(hostname)"
echo "--- numactl -H membind 行"; numactl -H 2>/dev/null | grep -E 'available|nodebind|membind'
echo "--- cpuset.mems / cpus"
for f in /sys/fs/cgroup/cpuset.mems /sys/fs/cgroup/cpuset.cpus; do
    echo "$f: $(cat $f 2>/dev/null)"
done
cat /proc/self/cpuset 2>/dev/null | while read c; do
    echo "cpuset=$c mems=$(cat /sys/fs/cgroup/cpuset$c/cpuset.mems 2>/dev/null) cpus=$(cat /sys/fs/cgroup/cpuset$c/cpuset.cpus 2>/dev/null)"
done
echo "--- 我分到的 cpu 属于哪个节点"
mycpu=$(cat /sys/fs/cgroup/cpuset.cpus 2>/dev/null || cat /sys/fs/cgroup/cpuset$(cat /proc/self/cpuset)/cpuset.cpus 2>/dev/null)
echo "cpus=$mycpu"
echo "--- 逐节点 --preferred 可用性（numactl 实测）"
for i in 0 4 7; do
    hbm=$((i+16))
    printf "node%d(cpu):%s  node%d(hbm)--preferred:" "$i" "$(numactl -N $i true 2>&1 >/dev/null && echo bindOK || echo bindFAIL)" "$hbm"
    numactl -N $i --preferred=$hbm true 2>&1 >/dev/null && echo OK || echo FAIL
done
