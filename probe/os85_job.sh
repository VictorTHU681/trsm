#!/usr/bin/env bash
# os85：OSW 筛查——owner 回写走线器化（对角内核 SWS 配速排水，flag 顺延一任务）
#   判读：store 桶应大降（排队 O(1)+下任务 sw_flush 残余），dkloop 微涨；task 净降 → 候选；净平/负 → 关闭
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S) $(git log --oneline -1)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
rm -f trsm_os85
gcc $FLAGS bench_trsm.c trsm.c -o trsm_os85 -lm -lkblas 2>&1 | grep -E "rror" | head -3
[[ -x trsm_os85 ]] || { echo "BUILD FAILED"; exit 1; }
echo "=== gen（OSW 组合）"
gcc -O3 -fopenmp -DTRSM_NO_SME_ASM -Dl_trsm=l_trsm_ref -c trsm.c -o probe/refo8.o 2>&1 | grep -E "error" || true
gcc $FLAGS probe/gen_test.c trsm.c probe/refo8.o -o probe/geno8 -lm 2>&1 | grep -E "error" || true
echo "[gen OSW owner30]";        TRSM_OSW=1 TRSM_OWNER=1 timeout 600 numactl -N 1 ./probe/geno8 30 2>&1 | tail -1
echo "[gen OSW steal20]";        TRSM_OSW=1 TRSM_OWNER=1 TRSM_STEAL=1 timeout 600 numactl -N 1 ./probe/geno8 20 2>&1 | tail -1
echo "[gen OSW cfold20]";        TRSM_OSW=1 TRSM_OWNER=1 TRSM_CFOLD=1 timeout 600 numactl -N 1 ./probe/geno8 20 2>&1 | tail -1
echo "[gen OSW 5thr20]";         TRSM_OSW=1 TRSM_OWNER=1 OMP_NUM_THREADS=5 timeout 600 numactl -N 1 ./probe/geno8 20 2>&1 | tail -1
run1(){ local lbl="$1"; shift; echo "--- $lbl"; env "$@" TRSM_PROFILE=2 OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./trsm_os85 512 19968 5 2>&1 | grep -E " x |PASS|FAIL|每线程平均"; }
echo "=== TC1 两线交错 x6"
for r in 1 2 3 4 5 6; do
  run1 "base_$r"
  run1 "osw_$r" TRSM_OSW=1
done
rm -f trsm_os85 probe/refo8.o probe/geno8
echo "=== DONE $(date +%H:%M:%S)"
