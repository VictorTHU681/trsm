#!/usr/bin/env bash
# 磨分 1：三用例新鲜分相基线 + TC1 owner 均衡 C0 扫描（尾部空闲 0.10ms 回收）
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S)"
. probe/env.sh >/dev/null 2>&1 || true
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
gcc $FLAGS bench_trsm.c trsm.c -o /tmp/trsm_g1 -lm -lkblas 2>&1 | grep -E "error" && { echo BUILD_FAIL; exit 1; }
export OMP_NUM_THREADS=38 OMP_PROC_BIND=close OMP_PLACES=cores OMP_WAIT_POLICY=active GOMP_SPINCOUNT=infinite
run1() { numactl -N 1 --preferred=17 /tmp/trsm_g1 "$@"; }
echo "--- 三用例分相基线（默认参数）"
for a in "512 19968" "2432 17024" "17024 512"; do
    TRSM_PROFILE=2 run1 $a 1 2>&1 | grep -E "512 x|2432 x|17024 x|每线程平均|唤醒" | tail -3
done
echo "--- TC1 C0 扫描（默认160；交错 2 轮，看时间与尾部空闲）"
for i in 1 2; do
    for c0 in 120 160 200 240; do
        t=$(TRSM_OWNER_C0=$c0 run1 512 19968 1 2>/dev/null | grep 512 | awk '{print $4}')
        tail=$(TRSM_OWNER_C0=$c0 TRSM_PROFILE=2 run1 512 19968 1 2>&1 | grep '每线程平均' | tail -1 | grep -oE '平均空闲 [0-9.]+' | awk '{print $2}')
        echo "C0=$c0 r$i: ${t}ms tail_idle=${tail}ms"
    done
done
echo "=== DONE $(date +%H:%M:%S)"
