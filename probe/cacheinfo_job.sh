#!/usr/bin/env bash
echo "=== HOST $(hostname) $(date +%H:%M:%S)"
echo "=== lscpu cache ==="; lscpu 2>/dev/null | grep -iE "cache|model name|CPU\(s\)|MHz"
echo "=== per-cpu cache (cpu0 as bound: cpu 38-75 on node1) ==="
for cpu in 0 38; do
 echo "-- cpu$cpu:"
 for c in /sys/devices/system/cpu/cpu$cpu/cache/index*; do
   printf "  L%s %s %s shared=%s\n" "$(cat $c/level)" "$(cat $c/type)" "$(cat $c/size)" "$(cat $c/shared_cpu_list)"
 done
done
echo "=== numactl -H (topology) ==="; numactl -H 2>/dev/null | head -8
echo "=== DONE"
