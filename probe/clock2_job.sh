#!/usr/bin/env bash
# cpufreq userspace 调速器：setspeed 归属与可写性（只读探测；写入若无权限会失败）
d=/sys/devices/system/cpu/cpu38/cpufreq
echo "host $(hostname)"
ls -l $d/ | grep -E "setspeed|governor|max_freq|cur_freq"
echo "setspeed=$(cat $d/scaling_setspeed 2>&1)"
echo "cpu0 (node0) governor=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor) cur=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq)"
echo "write test: $( (echo 2000000 > $d/scaling_setspeed) 2>&1 || true )"
echo "after: $(cat $d/scaling_cur_freq)"
cat /sys/devices/system/cpu/cpufreq/boost 2>/dev/null || echo "no boost knob"
