#!/usr/bin/env bash
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S)"
lscpu | grep -E "^CPU\(s\)|NUMA node|Thread|Core|Socket|L2|L3|Model name"
cat /sys/devices/system/node/node1/cpulist
echo "allowed under numactl -N 1: $(numactl -N 1 nproc)"; numactl -N 1 cat /proc/self/status | grep Cpus_allowed_list
echo "cgroup cpuset: $(cat /sys/fs/cgroup/cpuset/cpuset.cpus 2>/dev/null; cat /proc/self/cpuset 2>/dev/null)"
echo "--- busy processes (psr = cpu)"
ps -eLo pid,tid,psr,pcpu,comm --sort=-pcpu | awk '$4>5' | head -20
echo "--- per-cpu idle over 2s (node 1 cpus)"
mpstat -P ALL 2 1 2>/dev/null | awk 'NR>3 && $2 ~ /^[0-9]+$/ && $NF < 99.0 {print "cpu",$2,"idle",$NF}' | head -40
for c in $(seq 0 3); do cat /sys/devices/system/cpu/cpu$c/cache/index2/size 2>/dev/null; done | head -1
echo "L2 shared: $(cat /sys/devices/system/cpu/cpu32/cache/index2/shared_cpu_list 2>/dev/null) L3: $(cat /sys/devices/system/cpu/cpu32/cache/index3/size 2>/dev/null) $(cat /sys/devices/system/cpu/cpu32/cache/index3/shared_cpu_list 2>/dev/null)"
echo "=== DONE"
