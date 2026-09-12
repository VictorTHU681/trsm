#!/usr/bin/env bash
# TRSM_WAVE v1 正确性验证：全矩阵比对（nc256_dbg），WAVE on/off 对照，多形状含偏块/单面板
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S) $(git log --oneline -1)"
. probe/env.sh
FLAGS="-O2 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
rm -f probe/nc256_dbg
gcc $FLAGS probe/nc256_dbg.c trsm.c -o probe/nc256_dbg -lm 2>&1 | grep -E "rror" | head -5
[[ -x probe/nc256_dbg ]] || { echo COMPILE_FAIL; exit 1; }
for env in "TRSM_WAVE=1 TRSM_MB=256" "TRSM_MB=256"; do
  for mn in "2432 1024" "2432 2048" "2416 1024" "2432 128" "2048 896"; do
    echo "--- [$env] m n = $mn"
    env $env TRSM_OWNER=0 OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./probe/nc256_dbg $mn 2>&1 | head -6
  done
done
echo "=== DONE $(date +%H:%M:%S)"
