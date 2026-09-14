#!/usr/bin/env bash
# slx60：回写‖装载相位交错（TRSM_SLX，审计 #1）筛查
#   正确性：gen_test SLX=1 三配置（随机形状触发/绕过守卫边界）
#   性能：TC1 官方 bench base/SLX=1 交错 x6 + PROFILE=2 分相（预期 load 0.30→≤0.08、store 微增、task 净降；
#         上界 = store 全隐藏 −0.122 ms/线程 ≈ +6.5%，现实期望 +2~4%）
#   TC2/TC3 各 1 轮冒烟（守卫使 SLX 不触发，应与基线逐位一致）
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S) $(git log --oneline -1)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
rm -f trsm_s60
gcc $FLAGS bench_trsm.c trsm.c -o trsm_s60 -lm -lkblas 2>&1 | grep -E "rror" | head -5
[[ -x trsm_s60 ]] || { echo "BUILD FAILED"; exit 1; }
sha256sum trsm.c | cut -c1-40

echo "=== gen 正确性（SLX=1）"
gcc -O3 -fopenmp -DTRSM_NO_SME_ASM -Dl_trsm=l_trsm_ref -c trsm.c -o probe/ref60.o 2>&1 | grep -E "error" || true
gcc $FLAGS probe/gen_test.c trsm.c probe/ref60.o -o probe/gen_test60 -lm 2>&1 | grep -E "error" || true
echo "[gen SLX=1 40]";           TRSM_SLX=1 timeout 900 numactl -N 1 ./probe/gen_test60 40 2>&1 | tail -1
echo "[gen SLX=1 强制owner 30]"; TRSM_SLX=1 TRSM_OWNER=1 timeout 600 numactl -N 1 ./probe/gen_test60 30 2>&1 | tail -1
echo "[gen SLX=1 5线程 20]";     TRSM_SLX=1 OMP_NUM_THREADS=5 timeout 600 numactl -N 1 ./probe/gen_test60 20 2>&1 | tail -1

echo "=== TC1 两线交错 x6（官方 bench + PROFILE=2）"
run1(){ local lbl="$1"; shift; echo "--- $lbl"; env "$@" TRSM_PROFILE=2 OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./trsm_s60 512 19968 5 2>&1 | grep -E " x |PASS|FAIL|每线程平均|parallel"; }
for r in 1 2 3 4 5 6; do
  run1 "base_$r"
  run1 "slx_$r" TRSM_SLX=1
done

echo "=== TC2/TC3 冒烟（SLX=1，守卫应使其不触发）"
env TRSM_SLX=1 OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./trsm_s60 2432 17024 3 2>&1 | grep -E " x |PASS|FAIL"
env TRSM_SLX=1 OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./trsm_s60 17024 512 3 2>&1 | grep -E " x |PASS|FAIL"
rm -f probe/ref60.o probe/gen_test60
echo "=== DONE $(date +%H:%M:%S)"
