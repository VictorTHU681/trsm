#!/usr/bin/env bash
# LTO 专项 A/B（ctune 初筛 TC2 +1.9~2.3% 两轮/三轮）：同对二进制 base vs lto 交错，TC2 ×8、TC1/TC3 ×4
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S) $(git log --oneline -1)"
. probe/env.sh
uptime
BASE="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
rm -f t_base t_lto
gcc $BASE bench_trsm.c trsm.c -o t_base -lm -lkblas 2>&1 | grep -E "rror" | head -3 || true
gcc $BASE -mtune=tsv110 -flto bench_trsm.c trsm.c -o t_lto -lm -lkblas 2>&1 | grep -E "rror" | head -3 || true
[ -x t_base ] && [ -x t_lto ] || { echo "BUILD FAILED"; exit 1; }
echo "built: base $(md5sum t_base | cut -c1-8), lto $(md5sum t_lto | cut -c1-8)"
run1() { local b="$1" args="$2" lbl="$3"; echo "--- $lbl"
  OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT "./$b" $args 2>&1 | grep -E "PASS|FAIL"; }
for r in 1 2 3 4 5 6 7 8; do
  run1 t_base "2432 17024 1" "tc2_base_$r"
  run1 t_lto  "2432 17024 1" "tc2_lto_$r"
done
for r in 1 2 3 4; do
  run1 t_base "512 19968 1"  "tc1_base_$r"
  run1 t_lto  "512 19968 1"  "tc1_lto_$r"
  run1 t_base "17024 512 1"  "tc3_base_$r"
  run1 t_lto  "17024 512 1"  "tc3_lto_$r"
done
echo "=== DONE $(date +%H:%M:%S)"
