#!/usr/bin/env bash
# kd9：o9 诊断（审计 §10w 候选 4）——A 真流下 {ldnt,ld1d}×{软预取,无} 四变体，判 L2 硬件预取器能否接管 A 流
#   判读：o9d(ld1d 无预取) ≈ o9a(ldnt+软预取) → HW 预取器可带流（对手结构猜想成立，带宽化存在，
#         OANT −10% 改归因 L2 污染，候选 3 值得 PMU 重做）；o9d 明显更慢 → 每核 DRAM 行成本是硬约束，只剩减字节
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S) $(git log --oneline -1)"
. probe/env.sh
gcc -O2 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp -DTRSM_KDUP_LD1D probe/kdup2.c -o probe/kdup2_o9 -lm 2>&1 | grep -E "rror" | head -5
[[ -x probe/kdup2_o9 ]] || { echo "BUILD FAILED"; exit 1; }
HBM="${MEMOPT#--preferred=}"
echo "=== run1（单核 taskset）"
numactl -N 1 --preferred="$HBM" taskset -c 40 ./probe/kdup2_o9
echo "=== run2（复测）"
numactl -N 1 --preferred="$HBM" taskset -c 41 ./probe/kdup2_o9
rm -f probe/kdup2_o9
echo "=== DONE $(date +%H:%M:%S)"
