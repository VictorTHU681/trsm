#!/usr/bin/env bash
# explore-quant：官方 bench A/B（自动面板驻留 vs 关闭）各 3 轮 + 预取提示变体
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
rm -f trsm_test; gcc $FLAGS bench_trsm.c trsm.c -o trsm_test -lm -lkblas 2>&1 | grep -E "rror" | head -5
echo "=== 官方 bench TC1：自动（owner） ×3 / OWNER=0 ×3（题面原样命令）"
for r in 1 2 3; do OMP_NUM_THREADS=38 numactl -N 1 ./trsm_test 512 19968 1 2>&1 | grep -E "PASS|FAIL"; done
for r in 1 2 3; do TRSM_OWNER=0 OMP_NUM_THREADS=38 numactl -N 1 ./trsm_test 512 19968 1 2>&1 | grep -E "PASS|FAIL"; done
echo "=== 官方 bench TC2/TC3 ×2"
for r in 1 2; do for args in "2432 17024 1" "17024 512 1"; do OMP_NUM_THREADS=38 numactl -N 1 ./trsm_test $args 2>&1 | grep -E "PASS|FAIL"; done; done
echo "=== 预取提示变体（fastbench TC1）"
rm -f fastbench fastbench_l2 fastbench_ld
gcc $FLAGS probe/fastbench.c trsm.c -o fastbench -lm 2>&1 | grep -E "rror" | head -3
gcc $FLAGS -DTRSM_PF_HINT=pldl2keep probe/fastbench.c trsm.c -o fastbench_l2 -lm 2>&1 | grep -E "rror" | head -3
gcc $FLAGS -DTRSM_PF_LOAD=1 probe/fastbench.c trsm.c -o fastbench_ld -lm 2>&1 | grep -E "rror" | head -3
for b in fastbench fastbench_l2 fastbench_ld; do echo "--- [$b]"; [[ -x $b ]] && OMP_NUM_THREADS=38 numactl -N 1 ./$b -r 5 512 19968 2>&1 | grep -vE "^  rep"; done
echo "--- [fastbench_l2 OWNER=0]"; TRSM_OWNER=0 OMP_NUM_THREADS=38 numactl -N 1 ./fastbench_l2 -r 5 512 19968 2>&1 | grep -vE "^  rep"
echo "=== DONE $(date +%H:%M:%S)"
