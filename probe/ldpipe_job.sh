#!/usr/bin/env bash
# fresh29 LDPIPE v3：档1 = 回写循环里 prfm 下任务 B（NEXTA 同款节奏）；档2 = 旧边界 prfm+NT 拷贝（对照）
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S)"
. probe/env.sh >/dev/null 2>&1 || true
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
gcc $FLAGS bench_trsm.c trsm.c -o /tmp/trsm_lp -lm -lkblas 2>&1 | grep -E "error" && { echo BUILD_FAIL; exit 1; }
gcc $FLAGS probe/gen_test.c trsm.c probe/ref.o -o /tmp/gen_lp -lm 2>&1 | grep -E "error"
export OMP_NUM_THREADS=38 OMP_PROC_BIND=close OMP_PLACES=cores OMP_WAIT_POLICY=active GOMP_SPINCOUNT=infinite
echo "--- gen_test 20 档1 / 20 档2"
TRSM_LDPIPE=1 TRSM_MB=64 timeout 600 numactl -N 1 /tmp/gen_lp 20 2>&1 | tail -1
TRSM_LDPIPE=2 TRSM_MB=64 timeout 600 numactl -N 1 /tmp/gen_lp 20 2>&1 | tail -1
echo "--- TC1 A/B（off / 档1 / 档2，交错 4 轮）"
for i in 1 2 3 4; do
    for v in 0 1 2; do
        r=$(TRSM_LDPIPE=$v numactl -N 1 --preferred=17 /tmp/trsm_lp 512 19968 1 2>/dev/null | grep 512)
        echo "lp$v $i: $r"
    done
done
echo "--- 分相（off / 档1）"
for v in 0 1; do
    TRSM_LDPIPE=$v TRSM_PROFILE=2 numactl -N 1 --preferred=17 /tmp/trsm_lp 512 19968 1 2>&1 | grep -E "每线程平均" | tail -1
done
echo "=== DONE $(date +%H:%M:%S)"
