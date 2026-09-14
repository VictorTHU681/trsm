#!/usr/bin/env bash
# pkg64：v64 包自包含门禁——解包 + run.sh 原样执行
set -uo pipefail
W=$(mktemp -d /tmp/pkg64.XXXX)
unzip -q ~/S2/trsm_submit_v64_*.zip -d "$W" || { echo "UNZIP FAIL"; exit 1; }
cd "$W/trsm"
echo "=== HOST $(hostname) $(date +%H:%M:%S)"
bash run.sh 2>&1 | grep -E " x |PASS|FAIL|GFLOPS|编译|绑定|preferred|node"
echo "=== DONE $(date +%H:%M:%S)"
rm -rf "$W"
