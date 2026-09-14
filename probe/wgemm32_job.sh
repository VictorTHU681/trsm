#!/usr/bin/env bash
# W-GEMM v1 首轮验证：W 形状正确性（自然准入+强制小形状）、margin 回退路径、同二进制 on/off 三用例交错、分相
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S) $(git log --oneline -1)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
rm -f trsm_test
gcc $FLAGS bench_trsm.c trsm.c -o trsm_test -lm -lkblas 2>&1 | grep -E "rror" | head -5 || true
[ -x ./trsm_test ] || { echo "BUILD FAILED"; exit 1; }
sha256sum trsm.c
run1() { local lbl="$1" args="$2"; shift 2; printf "%-34s " "[$lbl]"; env "$@" OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./trsm_test $args 2>&1 | grep -E "PASS|FAIL" | head -1; }
echo "=== W 形状正确性（自然准入：m%64=0 n%96=0 npan≥2nth）"
run1 w_512x19968   "512 19968 1"
run1 w_1024x9984   "1024 9984 1"
run1 w_960x14592   "960 14592 1"
run1 w_64x7296     "64 7296 1"
run1 w_128x7392    "128 7392 1"
run1 w_448x7296    "448 7296 1"
echo "=== 强制 owner 小形状（TRSM_OWNER=1 MB=64 NC=96）"
run1 wf_64x96      "64 96 1"     TRSM_OWNER=1 TRSM_MB=64 TRSM_NC=96
run1 wf_128x192    "128 192 1"   TRSM_OWNER=1 TRSM_MB=64 TRSM_NC=96
run1 wf_256x480    "256 480 1"   TRSM_OWNER=1 TRSM_MB=64 TRSM_NC=96
run1 wf_512x960    "512 960 1"   TRSM_OWNER=1 TRSM_MB=64 TRSM_NC=96
echo "=== margin 强制回退（阈值拉满 → 走旧前向路径）"
run1 fb_512x19968  "512 19968 1" TRSM_WGEMM_MARGIN_MILLI=2000000000
run1 fb_1024x9984  "1024 9984 1" TRSM_WGEMM_MARGIN_MILLI=2000000000
echo "=== WGEMM=0 对照（旧路径完整性）"
run1 off_512x19968 "512 19968 1" TRSM_WGEMM=0
echo "=== 三用例 on/off 交错 ×3（判题口径）"
for r in 1 2 3; do
  run1 "tc1_off_$r" "512 19968 5"  TRSM_WGEMM=0
  run1 "tc1_on_$r"  "512 19968 5"
  run1 "tc2_on_$r"  "2432 17024 2"
  run1 "tc3_on_$r"  "17024 512 2"
done
echo "=== 分相（PROFILE=2）off/on"
env TRSM_WGEMM=0 TRSM_PROFILE=2 OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./trsm_test 512 19968 3 2>&1 | grep -E "每线程平均|parallel" | tail -2
env TRSM_PROFILE=2 OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./trsm_test 512 19968 3 2>&1 | grep -E "每线程平均|parallel" | tail -2
echo "=== 泛化回归（随机形状,多数不走 W 路径）"
gcc -O3 -fopenmp -DTRSM_NO_SME_ASM -Dl_trsm=l_trsm_ref -c trsm.c -o probe/ref.o 2>&1 | grep -E "error" || true
gcc $FLAGS probe/gen_test.c trsm.c probe/ref.o -o probe/gen_test -lm 2>&1 | grep -E "error" || true
echo "[gen 默认 30]"; timeout 600 numactl -N 1 ./probe/gen_test 30 2>&1 | tail -1
echo "[gen WGEMM=0 20]"; TRSM_WGEMM=0 timeout 600 numactl -N 1 ./probe/gen_test 20 2>&1 | tail -1
echo "[gen 强制owner 20]"; TRSM_OWNER=1 TRSM_MB=64 TRSM_NC=96 timeout 600 numactl -N 1 ./probe/gen_test 20 2>&1 | tail -1
echo "=== DONE $(date +%H:%M:%S)"
