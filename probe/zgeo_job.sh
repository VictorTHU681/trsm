#!/usr/bin/env bash
# fresh28 ZGEO v2：CPC（每调用 chunk 数）扫描——A 局部性 vs 排水稀释的权衡点
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S)"
. probe/env.sh >/dev/null 2>&1 || true
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
gcc $FLAGS bench_trsm.c trsm.c -o /tmp/trsm_zgeo -lm -lkblas 2>&1 | grep -E "error" && { echo BUILD_FAIL; exit 1; }
gcc $FLAGS probe/gen_test.c trsm.c probe/ref.o -o /tmp/gen_zgeo -lm 2>&1 | grep -E "error"
export OMP_NUM_THREADS=38 OMP_PROC_BIND=close OMP_PLACES=cores OMP_WAIT_POLICY=active GOMP_SPINCOUNT=infinite
echo "--- gen_test 20 CPC=1 / 20 CPC=2（块循环正确性）"
TRSM_ZGEO=1 TRSM_ZGEO_CPC=1 TRSM_MB=64 timeout 600 numactl -N 1 /tmp/gen_zgeo 20 2>&1 | tail -1
TRSM_ZGEO=1 TRSM_ZGEO_CPC=2 TRSM_MB=64 timeout 600 numactl -N 1 /tmp/gen_zgeo 20 2>&1 | tail -1
echo "--- A/B：off / CPC1 / CPC2 / CPC4 / 全融合，交错 2 轮"
for i in 1 2; do
    for v in off 1 2 4 0; do
        if [[ $v == off ]]; then env=""; else env="TRSM_ZGEO=1 TRSM_ZGEO_CPC=$v"; fi
        r=$(eval $env numactl -N 1 --preferred=17 /tmp/trsm_zgeo 512 19968 1 2>/dev/null | grep 512)
        echo "$v $i: $r"
    done
done
echo "--- 分相 kloop 桶（CPC=1 与 CPC=2）"
for v in 1 2; do
    TRSM_ZGEO=1 TRSM_ZGEO_CPC=$v TRSM_PROFILE=2 numactl -N 1 --preferred=17 /tmp/trsm_zgeo 512 19968 1 2>&1 | grep -E "每线程平均" | tail -1
done
echo "=== DONE $(date +%H:%M:%S)"
