#!/usr/bin/env bash
# 磨分 2：TC3 打包前瞻 LA 扫描（wait 0.93ms 回收）+ TC2 对角相细看（dkloop/ddump/depi 按 K 段分布）
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S)"
. probe/env.sh >/dev/null 2>&1 || true
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
gcc $FLAGS bench_trsm.c trsm.c -o /tmp/trsm_g2 -lm -lkblas 2>&1 | grep -E "error" && { echo BUILD_FAIL; exit 1; }
export OMP_NUM_THREADS=38 OMP_PROC_BIND=close OMP_PLACES=cores OMP_WAIT_POLICY=active GOMP_SPINCOUNT=infinite
run1() { numactl -N 1 --preferred=17 /tmp/trsm_g2 "$@"; }
echo "--- TC3 LA 扫描（默认2；交错 2 轮）"
for i in 1 2; do
    for la in 2 3 4 6; do
        t=$(TRSM_PACK_LA=$la run1 17024 512 1 2>/dev/null | grep 17024 | awk '{print $4}')
        w=$(TRSM_PACK_LA=$la TRSM_PROFILE=2 run1 17024 512 1 2>&1 | grep '每线程平均' | tail -1 | grep -oE 'wait [0-9.]+' | awk '{print $2}')
        echo "LA=$la r$i: ${t}ms wait=${w}ms"
    done
done
echo "--- TC2 基线复测（此节点校准）+ 对角相三桶"
for i in 1 2; do
    TRSM_PROFILE=2 run1 2432 17024 1 2>&1 | grep -E "2432 x|每线程平均" | tail -2
done
echo "=== DONE $(date +%H:%M:%S)"
