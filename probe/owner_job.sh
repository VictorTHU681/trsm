#!/usr/bin/env bash
# explore-quant 分支：面板驻留模式（TRSM_OWNER=1）正确性 + TC1 参数扫描 + 分相 profile
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
echo "=== 泛化正确性（owner=1，随机形状 m≤900）"
gcc -O3 -fopenmp -DTRSM_NO_SME_ASM -Dl_trsm=l_trsm_ref -c trsm.c -o probe/ref.o 2>&1 | grep -E "error"
gcc $FLAGS probe/gen_test.c trsm.c probe/ref.o -o probe/gen_test -lm 2>&1 | grep -E "error"
TRSM_OWNER=1 timeout 600 numactl -N 1 ./probe/gen_test 60 2>&1 | tail -1
TRSM_OWNER=1 TRSM_MB=128 TRSM_NC=96 timeout 600 numactl -N 1 ./probe/gen_test 40 2>&1 | tail -1
TRSM_OWNER=1 TRSM_MB=64 TRSM_NC=64 TRSM_KC=32 timeout 600 numactl -N 1 ./probe/gen_test 30 2>&1 | tail -1
TRSM_OWNER=1 TRSM_ANT=1 timeout 600 numactl -N 1 ./probe/gen_test 20 2>&1 | tail -1
timeout 600 numactl -N 1 ./probe/gen_test 20 2>&1 | tail -1

echo "=== TC1 扫描（fastbench -r 5，512×19968）"
rm -f fastbench; gcc $FLAGS probe/fastbench.c trsm.c -o fastbench -lm 2>&1 | grep -E "rror" | head -5; [[ -x fastbench ]] || { echo COMPILE_FAIL; exit 1; }
run1() { echo "--- [$*]"; env "$@" OMP_NUM_THREADS=38 numactl -N 1 ./fastbench -r 5 512 19968 2>&1 | grep -vE "^  rep"; }
run1 TRSM_MB=128 TRSM_NC=96 TRSM_KC=64
run1 TRSM_OWNER=1 TRSM_MB=128 TRSM_NC=96 TRSM_KC=64
run1 TRSM_OWNER=1 TRSM_MB=128 TRSM_NC=96 TRSM_KC=64 TRSM_ANT=1
run1 TRSM_OWNER=1 TRSM_MB=128 TRSM_NC=64 TRSM_KC=64
run1 TRSM_OWNER=1 TRSM_MB=128 TRSM_NC=128 TRSM_KC=64
run1 TRSM_OWNER=1 TRSM_MB=128 TRSM_NC=160 TRSM_KC=64
run1 TRSM_OWNER=1 TRSM_MB=128 TRSM_NC=96 TRSM_KC=128
run1 TRSM_OWNER=1 TRSM_MB=192 TRSM_NC=96 TRSM_KC=64
run1 TRSM_OWNER=1 TRSM_MB=192 TRSM_NC=128 TRSM_KC=64
run1 TRSM_OWNER=1 TRSM_MB=256 TRSM_NC=96 TRSM_KC=64
run1 TRSM_OWNER=1 TRSM_MB=256 TRSM_NC=64 TRSM_KC=64
run1 TRSM_OWNER=1 TRSM_MB=256 TRSM_NC=96 TRSM_KC=128
run1 TRSM_OWNER=1 TRSM_MB=64 TRSM_NC=96 TRSM_KC=64
run1 TRSM_OWNER=1 TRSM_MB=64 TRSM_NC=128 TRSM_KC=64
run1 TRSM_OWNER=1 TRSM_MB=128 TRSM_NC=96 TRSM_KC=64 TRSM_PF=0
run1 TRSM_OWNER=1 TRSM_MB=128 TRSM_NC=96 TRSM_KC=64 TRSM_NT=0

echo "=== TC1 分相 profile（官方 bench，TRSM_PROFILE=2）"
rm -f trsm_test; gcc $FLAGS bench_trsm.c trsm.c -o trsm_test -lm -lkblas 2>&1 | grep -E "rror" | head -5
prof1() { echo "--- [$*]"; env "$@" TRSM_PROFILE=2 OMP_NUM_THREADS=38 numactl -N 1 ./trsm_test 512 19968 1 2>&1 | grep -E "PASS|FAIL|prof\]" | tail -4; }
prof1 TRSM_OWNER=1 TRSM_MB=128 TRSM_NC=96 TRSM_KC=64
prof1 TRSM_OWNER=1 TRSM_MB=128 TRSM_NC=64 TRSM_KC=64
echo "=== DONE $(date +%H:%M:%S)"
