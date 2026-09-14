#!/usr/bin/env bash
# x1a（fresh35 worktree）：XONE 实验——chunk-0 旁流装载 vs 独立装载相
#   三线：bankbase（CURSOR=0）/ bankxone（CURSOR=0 XONE=1）/ cursor 生产参照
#   判读：xone 的 load 应 →~0.05（仅 i=0 任务），kloop 涨幅 < load 省幅 → 假说成立；kloop 爆涨 → MLP 池无余量，假说死
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S) $(git log --oneline -1) tree=$(pwd)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
rm -f trsm_x1a
gcc $FLAGS bench_trsm.c trsm.c -o trsm_x1a -lm -lkblas 2>&1 | grep -E "rror" | head -5
[[ -x trsm_x1a ]] || { echo "BUILD FAILED"; exit 1; }
echo "=== gen 正确性（XONE=1 bank 路径）"
gcc -O3 -fopenmp -DTRSM_NO_SME_ASM -Dl_trsm=l_trsm_ref -c trsm.c -o probe/refx1.o 2>&1 | grep -E "error" || true
gcc $FLAGS probe/gen_test.c trsm.c probe/refx1.o -o probe/gen_testx1 -lm 2>&1 | grep -E "error" || true
echo "[gen XONE 强制owner 30]"; TRSM_XONE=1 TRSM_OWNER=1 TRSM_OWNER_CURSOR=0 timeout 600 numactl -N 1 ./probe/gen_testx1 30 2>&1 | tail -1
echo "[gen XONE 5线程 20]";     TRSM_XONE=1 TRSM_OWNER=1 TRSM_OWNER_CURSOR=0 OMP_NUM_THREADS=5 timeout 600 numactl -N 1 ./probe/gen_testx1 20 2>&1 | tail -1
echo "[gen XONE+CFOLD 20]";     TRSM_XONE=1 TRSM_OWNER=1 TRSM_OWNER_CURSOR=0 TRSM_CFOLD=1 timeout 600 numactl -N 1 ./probe/gen_testx1 20 2>&1 | tail -1
run1(){ local lbl="$1"; shift; echo "--- $lbl"; env "$@" TRSM_PROFILE=2 OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./trsm_x1a 512 19968 5 2>&1 | grep -E " x |PASS|FAIL|每线程平均"; }
echo "=== TC1 三线交错 x6"
for r in 1 2 3 4 5 6; do
  run1 "bb_$r"  TRSM_OWNER_CURSOR=0
  run1 "bx_$r"  TRSM_OWNER_CURSOR=0 TRSM_XONE=1
  run1 "cur_$r"
done
rm -f trsm_x1a probe/refx1.o probe/gen_testx1
echo "=== DONE $(date +%H:%M:%S)"
