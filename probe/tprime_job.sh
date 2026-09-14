#!/usr/bin/env bash
# TPRIME 筛查：TLB 遍历引导（load/store/khead 行首 1 字节真实载入，prfm 藏不住页表遍历）
# ① 正确性（gen 40 + 官方三例 TPRIME 开） ② TC1 A/B 交错 ×6（off vs 16 vs 32；S=16 叠加）
# ③ PROFILE=2 分相定位（load/store/khead 应下降）
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S) $(git log --oneline -1)"
. probe/env.sh
uptime
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
rm -f trsm_test
gcc $FLAGS bench_trsm.c trsm.c -o trsm_test -lm -lkblas 2>&1 | grep -E "rror" | head -5 || true
[ -x ./trsm_test ] || { echo "BUILD FAILED"; exit 1; }
echo "=== 正确性"
gcc $FLAGS probe/gen_test.c trsm.c probe/ref.o -o probe/gen_test -lm 2>&1 | grep -E "rror" | head -3 || true
TRSM_TPRIME=16 TRSM_TPRIME_S=16 timeout 900 numactl -N 1 ./probe/gen_test 40 2>&1 | tail -1
for tc in "512 19968 1" "2432 17024 1" "17024 512 1"; do
  TRSM_TPRIME=16 TRSM_TPRIME_S=16 OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./trsm_test $tc 2>&1 | grep -E "PASS|FAIL"
done
echo "=== TC1 A/B 交错 ×6"
run1() { local lbl="$1"; shift; echo "--- $lbl"; env "$@" OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./trsm_test 512 19968 7 2>&1 | grep -E "PASS|FAIL"; }
for r in 1 2 3 4 5 6; do
  run1 "off_$r"      TRSM_TPRIME=0
  run1 "t16_$r"      TRSM_TPRIME=16
  run1 "t32_$r"      TRSM_TPRIME=32
  run1 "t16s16_$r"   TRSM_TPRIME=16 TRSM_TPRIME_S=16
  run1 "t16lp8_$r"   TRSM_TPRIME=16 TRSM_LOADPF=8    # 联动档：TLB 暖后 prfm 首次可能真生效（special5 loadpf 默认 0 疑因 TLB 冷行丢弃）
  run1 "t16lp16_$r"  TRSM_TPRIME=16 TRSM_LOADPF=16
  run1 "x1_$r"       TRSM_TPRIME_X=1                 # 跨任务：回写窗按行配速引导下一任务 B 页（纯写窗遍历器空闲）
  run1 "x1lp16_$r"   TRSM_TPRIME_X=1 TRSM_LOADPF=16
  run1 "x1t16lp16_$r" TRSM_TPRIME_X=1 TRSM_TPRIME=16 TRSM_LOADPF=16   # 全栈（首任务由 t16 兜底）
done
echo "=== TC2/TC3 抽查（t16s16 vs off 各 2）"
for r in 1 2; do
  for tc in "2432 17024 1" "17024 512 1"; do
    OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./trsm_test $tc 2>&1 | grep -E "PASS|FAIL"
    TRSM_TPRIME=16 TRSM_TPRIME_S=16 OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./trsm_test $tc 2>&1 | grep -E "PASS|FAIL"
  done
done
echo "=== 分相 off / t16s16 / x1lp16"
TRSM_PROFILE=2 OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./trsm_test 512 19968 5 2>&1 | grep "trsm prof" | tail -4
TRSM_TPRIME=16 TRSM_TPRIME_S=16 TRSM_PROFILE=2 OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./trsm_test 512 19968 5 2>&1 | grep "trsm prof" | tail -4
TRSM_TPRIME_X=1 TRSM_LOADPF=16 TRSM_PROFILE=2 OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./trsm_test 512 19968 5 2>&1 | grep "trsm prof" | tail -4
echo "=== DONE $(date +%H:%M:%S)"
