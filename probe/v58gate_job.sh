#!/usr/bin/env bash
# v58gate：合并树（CFOLD + DIAG_COMPACT + REHUGE + LTO + SHAPEP）联合门禁
#   run.sh×2 + 题面原样×2 + gen（默认80/CFOLD30/SWHOST40/JIT20/LTO40）+ 三用例三线交错 x3（全开 / 无CFOLD / 无DIAG_COMPACT）
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S) $(git log --oneline -1)"
echo "=== run.sh 第 1 轮"; bash ./run.sh 2>&1 | grep -E "PASS|FAIL|回退|rror"
echo "=== run.sh 第 2 轮"; bash ./run.sh 2>&1 | grep -E "PASS|FAIL"
. probe/env.sh
echo "=== 题面原样 x2"
for r in 1 2; do
  for args in "512 19968 1" "2432 17024 1" "17024 512 1"; do
    OMP_NUM_THREADS=38 numactl -N 1 ./trsm_test $args 2>&1 | grep -E "PASS|FAIL"
  done
done
echo "=== 泛化正确性"
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
gcc -O3 -fopenmp -DTRSM_NO_SME_ASM -Dl_trsm=l_trsm_ref -c trsm.c -o probe/ref.o 2>&1 | grep -E "error" || true
gcc $FLAGS probe/gen_test.c trsm.c probe/ref.o -o probe/gen_test -lm 2>&1 | grep -E "error" || true
timeout 900 numactl -N 1 ./probe/gen_test 80 2>&1 | tail -1
TRSM_CFOLD=1 TRSM_OWNER=1 timeout 600 numactl -N 1 ./probe/gen_test 30 2>&1 | tail -1
TRSM_DIAG_COMPACT=1 timeout 600 numactl -N 1 ./probe/gen_test 30 2>&1 | tail -1
TRSM_SWHOST=1 TRSM_MB=64 timeout 600 numactl -N 1 ./probe/gen_test 40 2>&1 | tail -1
TRSM_JIT=1 timeout 600 numactl -N 1 ./probe/gen_test 20 2>&1 | tail -1
gcc $FLAGS -mtune=tsv110 -flto probe/gen_test.c trsm.c probe/ref.o -o probe/gen_test_lto -lm 2>/dev/null \
  && timeout 600 numactl -N 1 ./probe/gen_test_lto 40 2>&1 | tail -1 || echo "LTO gen 编译失败"
echo "=== 三用例三线交错 x3（全开 / 无CFOLD / 无DIAG_COMPACT）"
run1(){ local lbl="$1"; shift; echo "--- $lbl"; env "$@" OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./trsm_test ${ARGS} 2>&1 | grep -E " x |PASS|FAIL"; }
for r in 1 2 3; do
  for args in "512 19968 5" "2432 17024 3" "17024 512 3"; do
    ARGS="$args" run1 "full_${args%% *}_$r"
    ARGS="$args" run1 "nocf_${args%% *}_$r" TRSM_CFOLD=0
    ARGS="$args" run1 "nodc_${args%% *}_$r" TRSM_DIAG_COMPACT=0
  done
done
echo "=== DONE $(date +%H:%M:%S)"
