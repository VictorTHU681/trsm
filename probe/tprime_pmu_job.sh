#!/usr/bin/env bash
# TPRIME 机制验证：DTLB 遍历计数 off vs x1lp16（ARM 标准事件 DTLB_WALK=0x34、L1D_TLB_REFILL=0x05、L2D_TLB_REFILL=0x2D）
# 预期：TPRIME 不减少遍历次数（遍历只是被提前、并发化），但 stall 下降、墙钟下降；
#       若 walk 数本来就少（当轮 X 覆盖率高），则本机看不出差异——先读覆盖率再解读。
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S) $(git log --oneline -1)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
rm -f trsm_test; gcc $FLAGS bench_trsm.c trsm.c -o trsm_test -lm -lkblas 2>&1 | grep -E "rror" | head -5
command -v perf >/dev/null || { echo NO_PERF; exit 1; }
echo "--- 当轮 X 页面状态"
TRSM_SMAPS=1 OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./trsm_test 512 19968 1 2>&1 | grep -E "smaps|thp|PASS" | head -8
for cfg in "off" "x1lp16"; do
  if [ "$cfg" = off ]; then ENVV=(TRSM_TPRIME_X=0); else ENVV=(TRSM_TPRIME_X=1 TRSM_LOADPF=16); fi
  echo "--- perf TC1 $cfg (3 轮)"
  for i in 1 2 3; do
    perf stat -e cycles,instructions,r0034,r0005,r002d \
      -- env "${ENVV[@]}" OMP_WAIT_POLICY=passive GOMP_SPINCOUNT=0 OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./trsm_test 512 19968 5 2>&1 | grep -E "PASS|cycles|instructions|r00|seconds" | head -10
  done
done
echo "=== DONE $(date +%H:%M:%S)"
