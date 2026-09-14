#!/usr/bin/env bash
# tc1p59：TC1 审计筛查捆绑（一次作业出三个判决）
#   #3 对角窗游标1消融：TRSM_DWPF=0 vs 基线 → load 不变=预取是死的（转 L2 提示版）；load 升且 dkloop 降=窗口已满（放弃）；load 升 dkloop 不动=有余量（转加密）
#   #4 A 游标提示层级：-DTRSM_AC1_L1（owner cursor 内核 ap 组 pldl2keep→pldl1keep）→ 看 kloop
#   #5 RFO 探针：load 相 cbuf 目标行若被 A 流逐出，全行覆盖写是否触发 write-allocate 读；B/A≥1.5 且 C≈A → dc zva 是 +4~6% 修法
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S) $(git log --oneline -1)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
rm -f trsm_p59 trsm_p59l1 rfoprobe
gcc $FLAGS bench_trsm.c trsm.c -o trsm_p59 -lm -lkblas 2>&1 | grep -E "rror" | head -3
gcc $FLAGS -DTRSM_AC1_L1 bench_trsm.c trsm.c -o trsm_p59l1 -lm -lkblas 2>&1 | grep -E "rror" | head -3
gcc -O3 -march=armv9-a+sve2 probe/rfoprobe.c -o rfoprobe 2>&1 | grep -E "rror" | head -3
[[ -x trsm_p59 && -x trsm_p59l1 && -x rfoprobe ]] || { echo "BUILD FAILED"; exit 1; }
sha256sum trsm.c | cut -c1-40

echo "=== #5 RFO 探针（单线程，HBM 首选）"
numactl -N 1 $MEMOPT ./rfoprobe

echo "=== #3/#4 in-app 三线交错 x4（TC1 官方 bench + PROFILE=2）"
run1(){ local lbl="$1" bin="$2"; shift 2; echo "--- $lbl"; env "$@" TRSM_PROFILE=2 OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./$bin 512 19968 5 2>&1 | grep -E " x |PASS|FAIL|trsm prof"; }
for r in 1 2 3 4; do
  run1 "base_$r"  trsm_p59
  run1 "dwpf0_$r" trsm_p59 TRSM_DWPF=0
  run1 "ac1l1_$r" trsm_p59l1
done
echo "=== DONE $(date +%H:%M:%S)"
