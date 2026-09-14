#!/usr/bin/env bash
# 编译档位筛查（账本确认从未扫过的维度）：-mtune=tsv110 / +LTO / +循环对齐 vs 现行 FLAGS
# 动机：手写 SME 内核不受影响，但 TC3 pack 引擎(3.06ms)与各例 load/派发循环是纯 C，
#       受 gcc 调度模型直接影响；判题机 gcc 12.3.1 支持 tsv110 流水线模型。
# 口径：官方 bench 判题口径（--preferred），3 轮 × 4 变体 × 3 用例，轮内变体轮转防节点漂移。
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S) $(git log --oneline -1)"
. probe/env.sh
uptime
BASE="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
declare -A FL=(
  [base]="$BASE"
  [mtune]="$BASE -mtune=tsv110"
  [lto]="$BASE -mtune=tsv110 -flto"
  [align]="$BASE -mtune=tsv110 -falign-functions=64 -falign-loops=32"
)
VS=(base mtune lto align)
for v in "${VS[@]}"; do
  rm -f "t_$v"
  gcc ${FL[$v]} bench_trsm.c trsm.c -o "t_$v" -lm -lkblas 2>&1 | grep -E "rror" | head -3 || true
  if [ -x "t_$v" ]; then echo "built $v: $(md5sum t_$v | cut -c1-8) $(stat -c%s t_$v)B"; else echo "BUILD FAIL $v"; fi
done
run1() { local v="$1" args="$2" lbl="$3"; [ -x "t_$v" ] || return 0
  echo "--- $lbl"
  OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT "./t_$v" $args 2>&1 | grep -E "PASS|FAIL"; }
for r in 1 2 3; do
  for i in 0 1 2 3; do
    v=${VS[$(( (i + r - 1) % 4 ))]}
    run1 "$v" "512 19968 1"  "tc1_${v}_$r"
    run1 "$v" "2432 17024 1" "tc2_${v}_$r"
    run1 "$v" "17024 512 1"  "tc3_${v}_$r"
  done
done
echo "=== DONE $(date +%H:%M:%S)"
