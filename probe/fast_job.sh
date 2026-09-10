#!/usr/bin/env bash
# 快速基准作业：fastbench 三用例（各 3 轮）+ 变体（每行一组 "K=V K=V"，来自 probe/fast_variants.txt）
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
rm -f fastbench; gcc $FLAGS probe/fastbench.c trsm.c -o fastbench -lm 2>&1 | grep -E "rror" | head -5; [[ -x fastbench ]] || { echo COMPILE_FAIL; exit 1; }
reps=${FAST_REPS:-3}
echo "--- 默认（题面原样 numactl -N 1）"
OMP_NUM_THREADS=38 numactl -N 1 ./fastbench -r $reps 2>&1 | grep -vE "^  rep"
if [[ -f probe/fast_variants.txt ]]; then
    while IFS= read -r v; do
        [[ -n "$v" && "$v" != \#* ]] || continue
        echo "--- [$v]"
        env $v OMP_NUM_THREADS=38 numactl -N 1 ./fastbench -r $reps 2>&1 | grep -vE "^  rep"
    done < probe/fast_variants.txt
fi
echo "=== DONE $(date +%H:%M:%S)"
