#!/usr/bin/env bash
# xr70：Xr 统一缓冲三道预筛门（TC2）
#   G1 写指令形态：xptm(st1d,默认) vs XPTM=0+XPNT=1(stnt1d) → store 桶差 ≥0.2ms = st1d 有暴露 RFO
#   G2 跳 Xp 写（诊断，校验必 FAIL 只读 prof）→ store 桶降幅 = Xr 天花板（门槛 −0.5ms）
#   G3 cbuf 冷环 N=2/4 → load/kloop 涨幅 ≤0.1ms = 冷块物理可接受
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S) $(git log --oneline -1)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
rm -f trsm_xr70
gcc $FLAGS bench_trsm.c trsm.c -o trsm_xr70 -lm -lkblas 2>&1 | grep -E "rror" | head -3
[[ -x trsm_xr70 ]] || { echo "BUILD FAILED"; exit 1; }
run2(){ local lbl="$1"; shift; echo "--- $lbl"; env "$@" TRSM_PROFILE=2 OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./trsm_xr70 2432 17024 5 2>&1 | grep -E " x |PASS|FAIL|每线程平均"; }
echo "=== 五线交错 x4"
for r in 1 2 3 4; do
  run2 "base_$r"
  run2 "g1nt_$r"  TRSM_XPTM=0 TRSM_XPNT=1
  run2 "g2sk_$r"  TRSM_G2SKIPXP=1
  run2 "g3r2_$r"  TRSM_G3RING=2
  run2 "g3r4_$r"  TRSM_G3RING=4
done
rm -f trsm_xr70
echo "=== DONE $(date +%H:%M:%S)"
