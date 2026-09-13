#!/usr/bin/env bash
# fresh29 LDPIPE v2：档 1（纯 prfm 进 L2）vs 档 2（prfm + 非临时拷贝限 2 行）vs off，交错 3 轮 + 分相
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S)"
. probe/env.sh >/dev/null 2>&1 || true
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
gcc $FLAGS bench_trsm.c trsm.c -o /tmp/trsm_lp -lm -lkblas 2>&1 | grep -E "error" && { echo BUILD_FAIL; exit 1; }
gcc $FLAGS probe/gen_test.c trsm.c probe/ref.o -o /tmp/gen_lp -lm 2>&1 | grep -E "error"
export OMP_NUM_THREADS=38 OMP_PROC_BIND=close OMP_PLACES=cores OMP_WAIT_POLICY=active GOMP_SPINCOUNT=infinite
echo "--- gen_test 30 档2（拷贝路径正确性）"
TRSM_LDPIPE=2 TRSM_MB=64 timeout 900 numactl -N 1 /tmp/gen_lp 30 2>&1 | tail -1
echo "--- TC1 A/B（off / 档1 / 档2，交错 3 轮）"
for i in 1 2 3; do
    for v in 0 1 2; do
        r=$(TRSM_LDPIPE=$v numactl -N 1 --preferred=17 /tmp/trsm_lp 512 19968 1 2>/dev/null | grep 512)
        echo "lp$v $i: $r"
    done
done
echo "--- 分相（档1 / 档2）"
for v in 1 2; do
    TRSM_LDPIPE=$v TRSM_PROFILE=2 numactl -N 1 --preferred=17 /tmp/trsm_lp 512 19968 1 2>&1 | grep -E "每线程平均" | tail -1
done
echo "=== DONE $(date +%H:%M:%S)"
