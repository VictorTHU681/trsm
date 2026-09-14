#!/usr/bin/env bash
# cf80：CFOLD-TC2 筛查——尾部 2 列奇面板折进面板 0（预期 +0.4~0.5%：消 10 任务整读 A 流 + 通用路径对角）
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S) $(git log --oneline -1)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
rm -f trsm_cf80
gcc $FLAGS bench_trsm.c trsm.c -o trsm_cf80 -lm -lkblas 2>&1 | grep -E "rror" | head -3
[[ -x trsm_cf80 ]] || { echo "BUILD FAILED"; exit 1; }
echo "=== gen（折叠全域回归）"
gcc -O3 -fopenmp -DTRSM_NO_SME_ASM -Dl_trsm=l_trsm_ref -c trsm.c -o probe/refc8.o 2>&1 | grep -E "error" || true
gcc $FLAGS probe/gen_test.c trsm.c probe/refc8.o -o probe/gen_c8 -lm 2>&1 | grep -E "error" || true
echo "[gen 默认 30]"; timeout 700 numactl -N 1 ./probe/gen_c8 30 2>&1 | tail -1
echo "[gen 大n非owner 20]"; TRSM_OWNER=0 timeout 700 numactl -N 1 ./probe/gen_c8 20 2>&1 | tail -1
run2(){ local lbl="$1"; shift; echo "--- $lbl"; env "$@" OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./trsm_cf80 2432 17024 5 2>&1 | grep -E " x |PASS|FAIL"; }
echo "=== TC2 两线 ABBA x6"
for r in 1 2 3; do
  run2 "off_a$r" TRSM_CFOLD=0
  run2 "on_a$r"
  run2 "on_b$r"
  run2 "off_b$r" TRSM_CFOLD=0
done
echo "=== TC1/TC3 守护"
env OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./trsm_cf80 512 19968 3 2>&1 | grep -E " x |PASS|FAIL"
env OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./trsm_cf80 17024 512 3 2>&1 | grep -E " x |PASS|FAIL"
rm -f trsm_cf80 probe/refc8.o probe/gen_c8
echo "=== DONE $(date +%H:%M:%S)"
