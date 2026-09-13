#!/usr/bin/env bash
# fresh28 ZGEO v3：NODRAIN 诊断——分离"8 瓦片纯算地板"与"排水管道成本"；off 分相对照
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S)"
. probe/env.sh >/dev/null 2>&1 || true
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
gcc $FLAGS bench_trsm.c trsm.c -o /tmp/trsm_zgeo -lm -lkblas 2>&1 | grep -E "error" && { echo BUILD_FAIL; exit 1; }
export OMP_NUM_THREADS=38 OMP_PROC_BIND=close OMP_PLACES=cores OMP_WAIT_POLICY=active GOMP_SPINCOUNT=infinite
echo "--- 分相对照：off / CPC1 / CPC1+NODRAIN / 全融+NODRAIN（NODRAIN 结果作废只看 kloop）"
echo "== off"
TRSM_PROFILE=2 numactl -N 1 --preferred=17 /tmp/trsm_zgeo 512 19968 1 2>&1 | grep -E "512|每线程平均" | tail -2
echo "== CPC1"
TRSM_ZGEO=1 TRSM_ZGEO_CPC=1 TRSM_PROFILE=2 numactl -N 1 --preferred=17 /tmp/trsm_zgeo 512 19968 1 2>&1 | grep -E "512|每线程平均" | tail -2
echo "== CPC1+NODRAIN"
TRSM_ZGEO=1 TRSM_ZGEO_CPC=1 TRSM_ZGEO_NODRAIN=1 TRSM_PROFILE=2 numactl -N 1 --preferred=17 /tmp/trsm_zgeo 512 19968 1 2>&1 | grep -E "512|每线程平均" | tail -2
echo "== 全融+NODRAIN"
TRSM_ZGEO=1 TRSM_ZGEO_CPC=0 TRSM_ZGEO_NODRAIN=1 TRSM_PROFILE=2 numactl -N 1 --preferred=17 /tmp/trsm_zgeo 512 19968 1 2>&1 | grep -E "512|每线程平均" | tail -2
echo "=== DONE $(date +%H:%M:%S)"
