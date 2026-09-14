#!/usr/bin/env bash
# st72：STEAL 面板窃取筛查——干净节点应 ≈0 成本；注入 2 核自旋扰动后 base 应大幅劣化、steal 应恢复大半
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S) $(git log --oneline -1)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
rm -f trsm_st72
gcc $FLAGS bench_trsm.c trsm.c -o trsm_st72 -lm -lkblas 2>&1 | grep -E "rror" | head -3
[[ -x trsm_st72 ]] || { echo "BUILD FAILED"; exit 1; }
echo "=== gen 正确性（STEAL=1 竞态覆盖）"
gcc -O3 -fopenmp -DTRSM_NO_SME_ASM -Dl_trsm=l_trsm_ref -c trsm.c -o probe/ref72.o 2>&1 | grep -E "error" || true
gcc $FLAGS probe/gen_test.c trsm.c probe/ref72.o -o probe/gen_test72 -lm 2>&1 | grep -E "error" || true
echo "[gen STEAL 强制owner 30]";      TRSM_STEAL=1 TRSM_OWNER=1 timeout 600 numactl -N 1 ./probe/gen_test72 30 2>&1 | tail -1
echo "[gen STEAL 5线程 20]";          TRSM_STEAL=1 TRSM_OWNER=1 OMP_NUM_THREADS=5 timeout 600 numactl -N 1 ./probe/gen_test72 20 2>&1 | tail -1
echo "[gen STEAL+CFOLD 20]";          TRSM_STEAL=1 TRSM_OWNER=1 TRSM_CFOLD=1 timeout 600 numactl -N 1 ./probe/gen_test72 20 2>&1 | tail -1
run1(){ local lbl="$1"; shift; echo "--- $lbl"; env "$@" OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./trsm_st72 512 19968 5 2>&1 | grep -E " x |PASS|FAIL"; }
echo "=== 干净两线交错 x6"
for r in 1 2 3 4 5 6; do
  run1 "cl_base_$r"
  run1 "cl_st_$r" TRSM_STEAL=1
done
echo "=== 扰动两线交错 x5（自旋孤儿钉 node1 两核）"
CPUS=$(numactl -H 2>/dev/null | awk '/node 1 cpus/{print $4, $5}')
C1=$(echo $CPUS | awk '{print $1}'); C2=$(echo $CPUS | awk '{print $2}')
echo "hog cores: $C1 $C2"
timeout 130 taskset -c $C1 bash -c 'while :; do :; done' &
timeout 130 taskset -c $C2 bash -c 'while :; do :; done' &
sleep 1
for r in 1 2 3 4 5; do
  run1 "pt_base_$r"
  run1 "pt_st_$r" TRSM_STEAL=1
done
wait || true
rm -f trsm_st72 probe/ref72.o probe/gen_test72
echo "=== DONE $(date +%H:%M:%S)"
