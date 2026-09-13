#!/usr/bin/env bash
# REHUGE 收益测量（fresh27）：{自然态, 人工碎片态} × {REHUGE=0, 1} 交错，
# 记录 buddyinfo（节点 order≥9 空闲块）、X 覆盖率（TRSM_SMAPS）、TC1 fastbench 与官方 bench。
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
rm -f fastbench; gcc $FLAGS probe/fastbench.c trsm.c -o fastbench -lm 2>&1 | grep -E "rror" | head -3
gcc -O2 -o probe/fragger probe/fragger.c 2>&1 | grep -E "rror" | head -3
gcc $FLAGS bench_trsm.c trsm.c -o trsm_test -lm -lkblas 2>&1 | grep -E "rror" | head -3
[[ -x fastbench && -x probe/fragger && -x trsm_test ]] || { echo COMPILE_FAIL; exit 1; }

# HBM 节点（与 run.sh 同逻辑）
cpunode=1; hbm=""
read -ra dist < /sys/devices/system/node/node$cpunode/distance
best=999
for n in "${!dist[@]}"; do
    [[ -d /sys/devices/system/node/node$n ]] || continue
    [[ -z "$(cat /sys/devices/system/node/node$n/cpulist 2>/dev/null)" ]] || continue
    (( dist[n] < best )) && { best=${dist[n]}; hbm=$n; }
done
echo "hbm=$hbm"

buddy() { grep "Node $hbm," /proc/buddyinfo; grep "MemFree" /sys/devices/system/node/node$hbm/meminfo | tr -s " "; }

phase() {   # $1=标签
    echo "===== 阶段 [$1] $(date +%H:%M:%S)"
    echo "--- buddyinfo:"; buddy
    echo "--- 覆盖率检查（REHUGE=0，单轮，带 SMAPS）:"
    TRSM_REHUGE=0 TRSM_SMAPS=1 OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./fastbench -r 1 512 19968 2>&1 | grep -E "smaps|rehuge|PASS|FAIL" | head -8
    echo "--- 覆盖率检查（REHUGE=1，单轮，带 SMAPS）:"
    TRSM_REHUGE=1 TRSM_SMAPS=1 OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./fastbench -r 1 512 19968 2>&1 | grep -E "smaps|rehuge|PASS|FAIL" | head -8
    echo "--- TC1 计时（fastbench -r 5，交错 3 对，无 SMAPS）:"
    for i in 1 2 3; do
        for r in 0 1; do
            out=$(TRSM_REHUGE=$r OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./fastbench -r 5 512 19968 2>&1 | grep " x ")
            echo "R$r #$i: $out"
        done
    done
    echo "--- TC1 官方 bench（各 1 轮）:"
    for r in 0 1; do
        out=$(TRSM_REHUGE=$r OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./trsm_test 512 19968 1 2>/dev/null | grep -E "512")
        echo "R$r: $out"
    done
    echo "--- TC2/TC3 官方 bench（REHUGE=1 一轮，正确性 + 副作用检查）:"
    TRSM_REHUGE=1 OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./trsm_test 2432 17024 1 2>/dev/null | grep -E "2432"
    TRSM_REHUGE=1 OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./trsm_test 17024 512 1 2>/dev/null | grep -E "17024"
}

phase "自然态"

echo "===== 碎片器 v2：全毒化（keep=16/2MB, pct=100，容量保留 ~3.5GB）"
probe/fragger $hbm 3900 1800 16 100 2> logs/fragger_a_$$.log &
FR=$!
sleep 15
head -8 logs/fragger_a_$$.log
phase "全毒化"
kill $FR 2>/dev/null; wait $FR 2>/dev/null; sleep 3

echo "===== 碎片器 v2：半毒化（keep=1/2MB, pct=55——覆盖率旋钮中档）"
probe/fragger $hbm 3900 1800 1 55 2> logs/fragger_b_$$.log &
FR=$!
sleep 15
head -8 logs/fragger_b_$$.log
phase "半毒化"
kill $FR 2>/dev/null; wait $FR 2>/dev/null; sleep 3

echo "===== 碎片器退出后 buddyinfo:"; buddy
echo "=== DONE $(date +%H:%M:%S)"
