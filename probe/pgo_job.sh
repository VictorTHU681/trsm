#!/usr/bin/env bash
# PGO 筛查（LTO 已证编译器敏感）：base vs lto vs lto+pgo（小形状训练），TC1 ×6 + TC2/3 ×2 交错
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S) $(git log --oneline -1)"
. probe/env.sh
uptime
BASE="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
LTO="-mtune=tsv110 -flto"
rm -f t_base t_lto t_pgo; rm -rf pgod; mkdir -p pgod
gcc $BASE bench_trsm.c trsm.c -o t_base -lm -lkblas 2>&1 | grep -E "rror" | head -3 || true
gcc $BASE $LTO bench_trsm.c trsm.c -o t_lto -lm -lkblas 2>&1 | grep -E "rror" | head -3 || true
gcc $BASE $LTO -fprofile-generate -fprofile-dir=./pgod bench_trsm.c trsm.c -o t_pgen -lm -lkblas 2>&1 | grep -E "rror" | head -3 || true
if [ -x t_pgen ]; then
  echo "--- PGO 训练（512x1920 owner 路径 + 2432x1664 special4 路径，输出丢弃）"
  OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./t_pgen 512 1920 1  > /dev/null 2>&1 || true
  OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./t_pgen 2432 1664 1 > /dev/null 2>&1 || true
  gcc $BASE $LTO -fprofile-use -fprofile-dir=./pgod -fprofile-correction bench_trsm.c trsm.c -o t_pgo -lm -lkblas 2>&1 | grep -E "rror" | head -3 || true
fi
for v in base lto pgo; do [ -x "t_$v" ] && echo "built $v: $(md5sum t_$v | cut -c1-8)" || echo "MISS $v"; done
run1() { local b="$1" args="$2" lbl="$3"; [ -x "$b" ] || return 0; echo "--- $lbl"
  OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT "./$b" $args 2>&1 | grep -E "PASS|FAIL"; }
for r in 1 2 3 4 5 6; do
  run1 t_base "512 19968 1" "tc1_base_$r"
  run1 t_lto  "512 19968 1" "tc1_lto_$r"
  run1 t_pgo  "512 19968 1" "tc1_pgo_$r"
done
for r in 1 2; do
  run1 t_lto  "2432 17024 1" "tc2_lto_$r"
  run1 t_pgo  "2432 17024 1" "tc2_pgo_$r"
  run1 t_lto  "17024 512 1"  "tc3_lto_$r"
  run1 t_pgo  "17024 512 1"  "tc3_pgo_$r"
done
echo "=== DONE $(date +%H:%M:%S)"
