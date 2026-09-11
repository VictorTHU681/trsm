#!/usr/bin/env bash
# explore-quant：case2 / case3 调度参数重扫（v22 内核之后未再验证）+ 部分和缓冲放置
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
rm -f fastbench; gcc $FLAGS probe/fastbench.c trsm.c -o fastbench -lm 2>&1 | grep -E "rror" | head -5; [[ -x fastbench ]] || { echo COMPILE_FAIL; exit 1; }
r3() { echo "--- [$*]"; env "$@" OMP_NUM_THREADS=38 numactl -N 1 ./fastbench -r 4 17024 512 2>&1 | grep -vE "^  rep"; }
r2() { echo "--- [$*]"; env "$@" OMP_NUM_THREADS=38 numactl -N 1 ./fastbench -r 4 2432 17024 2>&1 | grep -vE "^  rep"; }
echo "=== case3"
r3 A=1
r3 TRSM_PARTNODE=-1
r3 TRSM_TAIL_ROWS=4 TRSM_TAIL_SEG=10 TRSM_MAXSEG=10
r3 TRSM_TAIL_ROWS=3 TRSM_TAIL_SEG=8 TRSM_MAXSEG=8
r3 TRSM_TAIL_ROWS=6 TRSM_TAIL_SEG=10 TRSM_MAXSEG=12
r3 TRSM_TAIL_ROWS=4 TRSM_TAIL_SEG=8
r3 TRSM_SPLITF=8
r3 TRSM_SPLITF=16
r3 TRSM_SPLITF=24 TRSM_MAXSEG=12
r3 TRSM_PACK_LA=1
r3 TRSM_PACK_LA=3
r3 TRSM_PACK_LA=4
r3 TRSM_SWHOST=0
r3 TRSM_RESERVE=0
echo "=== case2"
r2 A=1
r2 TRSM_PARTNODE=-1
r2 TRSM_SPLITF=8
r2 TRSM_SPLITF=16
r2 TRSM_SPLITF=0
r2 TRSM_MAXSEG=4
r2 TRSM_LSPLIT=0
r2 TRSM_XPNT=1
r2 TRSM_NT=0
r2 TRSM_ANT=0
r2 TRSM_SWHOST=1
r2 TRSM_MB=192 TRSM_NC=160
r2 TRSM_MB=160 TRSM_NC=128 TRSM_KC=32
echo "=== DONE $(date +%H:%M:%S)"
