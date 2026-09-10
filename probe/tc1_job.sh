#!/usr/bin/env bash
# explore-quant 分支作业：(1) 低精度外积吞吐探针；(2) TC1 块参数扫描（fastbench 只跑 512×19968）；(3) TC1 分相 profile（官方 bench）
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"

echo "=== 低精度外积吞吐（寄存器内轮转，无访存）"
if gcc -O2 -march=armv9-a+sme+sme-f64f64+sme-i16i64+sve2 -fopenmp -DHAVE_I16I64 probe/lowprec_peak.c -o probe/lowprec_peak 2>/dev/null; then :
elif gcc -O2 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp probe/lowprec_peak.c -o probe/lowprec_peak 2>&1 | grep -E "rror" | head -5; then echo "LOWPREC_COMPILE_FAIL"; fi
[[ -x probe/lowprec_peak ]] && OMP_NUM_THREADS=38 numactl -N 1 ./probe/lowprec_peak

echo "=== TC1 块参数扫描（fastbench -r 5，只跑 512×19968，min/avg）"
rm -f fastbench; gcc $FLAGS probe/fastbench.c trsm.c -o fastbench -lm 2>&1 | grep -E "rror" | head -5; [[ -x fastbench ]] || { echo COMPILE_FAIL; exit 1; }
run1() { echo "--- [$*]"; env "$@" OMP_NUM_THREADS=38 numactl -N 1 ./fastbench -r 5 512 19968 2>&1 | grep -vE "^  rep"; }
run1 A=1
run1 TRSM_MB=128 TRSM_NC=96 TRSM_KC=64
run1 TRSM_MB=128 TRSM_NC=64 TRSM_KC=64
run1 TRSM_MB=128 TRSM_NC=128 TRSM_KC=64
run1 TRSM_MB=128 TRSM_NC=160 TRSM_KC=64
run1 TRSM_MB=128 TRSM_NC=96 TRSM_KC=32
run1 TRSM_MB=96 TRSM_NC=96 TRSM_KC=32
run1 TRSM_MB=96 TRSM_NC=128 TRSM_KC=32
run1 TRSM_MB=64 TRSM_NC=96 TRSM_KC=64
run1 TRSM_MB=64 TRSM_NC=128 TRSM_KC=64
run1 TRSM_MB=64 TRSM_NC=192 TRSM_KC=64
run1 TRSM_MB=64 TRSM_NC=256 TRSM_KC=64
run1 TRSM_MB=256 TRSM_NC=96 TRSM_KC=64
run1 TRSM_MB=128 TRSM_NC=96 TRSM_KC=64 TRSM_ANT=1
run1 TRSM_MB=128 TRSM_NC=96 TRSM_KC=64 TRSM_SWHOST=1
run1 TRSM_MB=128 TRSM_NC=96 TRSM_KC=64 TRSM_RESERVE=0
run1 TRSM_MB=128 TRSM_NC=96 TRSM_KC=64 TRSM_XPNODE=-2
run1 TRSM_MB=128 TRSM_NC=96 TRSM_KC=64 TRSM_PF=0

echo "=== TC1 分相 profile（官方 bench，TRSM_PROFILE=2）"
rm -f trsm_test; gcc $FLAGS bench_trsm.c trsm.c -o trsm_test -lm -lkblas 2>&1 | grep -E "rror" | head -5
prof1() { echo "--- [$*]"; env "$@" TRSM_PROFILE=2 OMP_NUM_THREADS=38 numactl -N 1 ./trsm_test 512 19968 1 2>&1 | grep -E "PASS|FAIL|prof\]" | tail -4; }
prof1 A=1
prof1 TRSM_MB=128 TRSM_NC=96 TRSM_KC=64
prof1 TRSM_MB=64 TRSM_NC=128 TRSM_KC=64
echo "=== DONE $(date +%H:%M:%S)"
