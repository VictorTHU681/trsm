#!/usr/bin/env bash
# v57 提交包端到端验收：解压 → bash run.sh → 3 PASS；确认节点选择行为（node7 或回退）
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S)"
echo "--- 本机 NUMA 拓扑（前 3 行）"; numactl -H 2>/dev/null | head -3
W=/tmp/trsm_pkgv57_$$
rm -rf "$W"; mkdir -p "$W"
ls ~/S2/trsm_submit_v57_*.zip || { echo "NO ZIP"; exit 1; }
unzip -q ~/S2/trsm_submit_v57_*.zip -d "$W" || { echo "UNZIP FAIL"; exit 1; }
cd "$W/trsm" || { echo "CD FAIL"; exit 1; }
echo "=== run.sh start $(date +%H:%M:%S)"
timeout -k 10 900 bash run.sh 2> err.txt | tee out.txt
rc1=$?
echo "=== run.sh 第 2 轮 $(date +%H:%M:%S)"
timeout -k 10 900 bash run.sh 2>> err.txt | tee -a out.txt
rc=$?
echo "=== run.sh exit=$rc $(date +%H:%M:%S)"
echo "--- 节点选择（run.sh stderr）"; grep -E '\[run.sh\]' err.txt
echo "--- PASS 行数（期望 3（×2 轮=6））"; grep -c 'PASS' out.txt
cd /; rm -rf "$W"
echo "=== DONE $(date +%H:%M:%S)"
