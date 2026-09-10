#!/usr/bin/env bash
# 计算节点时钟 / FMOPA 吞吐核实：cpufreq 策略与上限、负载下实际频率、单核与 38 核 FMOPA 峰值
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S)"
. probe/env.sh
echo "=== CPU 型号"
lscpu | grep -E "Model name|CPU max MHz|CPU min MHz|CPU MHz|Thread|Core|Socket|NUMA node\(s\)" | head -12
grep -m1 -E "CPU implementer|CPU part|CPU variant|CPU revision" /proc/cpuinfo; grep -E "CPU (implementer|part|variant|revision)" /proc/cpuinfo | sort | uniq -c | head
echo "=== cpufreq（cpu38：node 1 首核）"
d=/sys/devices/system/cpu/cpu38/cpufreq
for f in scaling_driver scaling_governor scaling_available_governors scaling_min_freq scaling_max_freq cpuinfo_min_freq cpuinfo_max_freq scaling_cur_freq cpuinfo_cur_freq scaling_available_frequencies; do
    printf "%-30s %s\n" "$f" "$(cat $d/$f 2>/dev/null || echo n/a)"
done
echo "=== 各核 scaling_cur_freq 分布（空闲）"
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq 2>/dev/null | sort -n | uniq -c | head
echo "=== 负载下频率：后台 38 线程 FMOPA 峰值探针，同时采样 cpu38..75 频率"
gcc -O2 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp probe/peak.c -o probe/peak 2>&1 | grep -E "rror" | head -3
( OMP_NUM_THREADS=38 numactl -N 1 ./probe/peak 8 > /tmp/peak_out.txt 2>&1 ) &
pid=$!
sleep 1.5
for i in 1 2 3; do
    echo "--- 采样 $i: $(for c in $(seq 38 75); do cat /sys/devices/system/cpu/cpu$c/cpufreq/scaling_cur_freq 2>/dev/null; done | sort -n | uniq -c | tr '\n' ';')"
    sleep 0.5
done
wait $pid
grep -E "FMOPA|内核形态|threads" /tmp/peak_out.txt
echo "=== 计时器频率 cntfrq / 用 PMU 周期计数估算实际核频（perf 若可用）"
which perf >/dev/null 2>&1 && OMP_NUM_THREADS=1 taskset -c 40 perf stat -e cycles,instructions -x, ./probe/peak 1 2>&1 | grep -E "cycles|instructions|FMOPA.D 8-tile" | head -5 || echo "perf 不可用"
echo "=== DONE $(date +%H:%M:%S)"
