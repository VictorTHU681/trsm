#!/usr/bin/env bash
# explore-quant 分支：面板驻留模式（自动策略）正确性 + 三用例 + TC1 变体
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
echo "=== 泛化正确性（随机形状 m≤900：自动策略 + 强制 owner 各种块）"
gcc -O3 -fopenmp -DTRSM_NO_SME_ASM -Dl_trsm=l_trsm_ref -c trsm.c -o probe/ref.o 2>&1 | grep -E "error"
gcc $FLAGS probe/gen_test.c trsm.c probe/ref.o -o probe/gen_test -lm 2>&1 | grep -E "error"
timeout 600 numactl -N 1 ./probe/gen_test 60 2>&1 | tail -1
TRSM_OWNER=1 timeout 600 numactl -N 1 ./probe/gen_test 40 2>&1 | tail -1
TRSM_OWNER=1 TRSM_MB=128 TRSM_NC=64 timeout 600 numactl -N 1 ./probe/gen_test 20 2>&1 | tail -1
TRSM_OWNER=1 TRSM_MB=32 TRSM_NC=32 TRSM_KC=32 timeout 600 numactl -N 1 ./probe/gen_test 20 2>&1 | tail -1
TRSM_OWNER=1 OMP_NUM_THREADS=5 timeout 600 numactl -N 1 ./probe/gen_test 20 2>&1 | tail -1
TRSM_OWNER=0 timeout 600 numactl -N 1 ./probe/gen_test 20 2>&1 | tail -1
TRSM_JIT=1 timeout 600 numactl -N 1 ./probe/gen_test 20 2>&1 | tail -1

echo "=== 三用例（fastbench，自动策略：TC1 走面板驻留，TC2/3 不变）"
rm -f fastbench; gcc $FLAGS probe/fastbench.c trsm.c -o fastbench -lm 2>&1 | grep -E "rror" | head -5; [[ -x fastbench ]] || { echo COMPILE_FAIL; exit 1; }
OMP_NUM_THREADS=38 numactl -N 1 ./fastbench -r 3 2>&1 | grep -vE "^  rep"
echo "--- [TRSM_OWNER=0 基线]"
TRSM_OWNER=0 OMP_NUM_THREADS=38 numactl -N 1 ./fastbench -r 3 2>&1 | grep -vE "^  rep"

echo "=== TC1 变体（fastbench -r 5）"
run1() { echo "--- [$*]"; env "$@" OMP_NUM_THREADS=38 numactl -N 1 ./fastbench -r 5 512 19968 2>&1 | grep -vE "^  rep"; }
run1 A=1
run1 TRSM_NC=128
run1 TRSM_MB=128
run1 TRSM_LCOPIES=1
run1 TRSM_LCOPIES=2
rm -f fastbench_pad; gcc $FLAGS -DTRSM_NCPAD=8 probe/fastbench.c trsm.c -o fastbench_pad -lm 2>&1 | grep -E "rror" | head -5
run1p() { echo "--- [NCPAD=8 $*]"; env "$@" OMP_NUM_THREADS=38 numactl -N 1 ./fastbench_pad -r 5 512 19968 2>&1 | grep -vE "^  rep"; }
[[ -x fastbench_pad ]] && { run1p A=1; run1p TRSM_NC=128; run1p TRSM_NC=64; }

echo "=== 官方 bench 三用例（题面原样命令）"
rm -f trsm_test; gcc $FLAGS bench_trsm.c trsm.c -o trsm_test -lm -lkblas 2>&1 | grep -E "rror" | head -5
for args in "512 19968 1" "2432 17024 1" "17024 512 1"; do
    OMP_NUM_THREADS=38 numactl -N 1 ./trsm_test $args 2>&1 | grep -E "PASS|FAIL"
done
echo "=== DONE $(date +%H:%M:%S)"
