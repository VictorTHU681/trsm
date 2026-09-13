#!/usr/bin/env bash
# 诊断包端到端验收：解压 → 注入 bench → bash run.sh（流式输出 + 超时），检查五行 PASS + 诊断段完整
# diag3 = TC1 三方对照（tc1a 基线 / tc1b 压实器 / tc1c REHUGE）+ TC2/TC3 压实，共 5 次 bench
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S)"
W=/tmp/trsm_diagpkg_$$
rm -rf "$W"; mkdir -p "$W"
ls ~/S2/trsm_submit_diag3_*.zip || { echo "NO ZIP"; exit 1; }
unzip -q ~/S2/trsm_submit_diag3_*.zip -d "$W" || { echo "UNZIP FAIL"; exit 1; }
cp bench_trsm.c "$W/trsm/" || { echo "CP FAIL"; exit 1; }
cd "$W/trsm" || { echo "CD FAIL"; exit 1; }
echo "=== run.sh start $(date +%H:%M:%S)"
timeout -k 10 1800 bash run.sh 2> err.txt | tee out.txt
rc=$?
echo ""
echo "=== run.sh exit=$rc $(date +%H:%M:%S)"
echo "--- run.sh 自身 stderr（末 10 行）---"
tail -10 err.txt
echo "--- PASS 行数（期望 5）---"
grep -c 'PASS' out.txt
echo "--- 诊断段行数 ---"
awk '/\[trsm-diag\]/{f=1} f' out.txt | wc -l
cd /; rm -rf "$W"
echo "=== DONE $(date +%H:%M:%S)"
