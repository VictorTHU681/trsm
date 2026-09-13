#!/usr/bin/env bash
# fresh29 LDPIPE 冒烟：正确性（gen_test + TC1 校验）+ 同二进制 A/B 交错 4 对 + 分相
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S)"
. probe/env.sh >/dev/null 2>&1 || true
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
gcc $FLAGS bench_trsm.c trsm.c -o /tmp/trsm_lp -lm -lkblas 2>&1 | grep -E "error" && { echo BUILD_FAIL; exit 1; }
gcc $FLAGS probe/gen_test.c trsm.c probe/ref.o -o /tmp/gen_lp -lm 2>&1 | grep -E "error"
export OMP_NUM_THREADS=38 OMP_PROC_BIND=close OMP_PLACES=cores OMP_WAIT_POLICY=active GOMP_SPINCOUNT=infinite
echo "--- gen_test 40 LDPIPE=1 / 20 LDPIPE=1+MB64"
TRSM_LDPIPE=1 timeout 900 numactl -N 1 /tmp/gen_lp 40 2>&1 | tail -1
TRSM_LDPIPE=1 TRSM_MB=64 timeout 600 numactl -N 1 /tmp/gen_lp 20 2>&1 | tail -1
echo "--- TC1 A/B（同二进制，交错 4 对）"
for i in 1 2 3 4; do
    r0=$(numactl -N 1 --preferred=17 /tmp/trsm_lp 512 19968 1 2>/dev/null | grep 512)
    r1=$(TRSM_LDPIPE=1 numactl -N 1 --preferred=17 /tmp/trsm_lp 512 19968 1 2>/dev/null | grep 512)
    echo "off $i: $r0"; echo "on  $i: $r1"
done
echo "--- 分相（off 与 on 各一发，看 load/xcopy/kloop）"
TRSM_PROFILE=2 numactl -N 1 --preferred=17 /tmp/trsm_lp 512 19968 1 2>&1 | grep -E "每线程平均" | tail -1
TRSM_LDPIPE=1 TRSM_PROFILE=2 numactl -N 1 --preferred=17 /tmp/trsm_lp 512 19968 1 2>&1 | grep -E "每线程平均" | tail -1
echo "--- TC2/TC3 不受影响确认（owner 路径外）"
TRSM_LDPIPE=1 numactl -N 1 --preferred=17 /tmp/trsm_lp 2432 17024 1 2>/dev/null | grep -E "PASS|FAIL"
TRSM_LDPIPE=1 numactl -N 1 --preferred=17 /tmp/trsm_lp 17024 512 1 2>/dev/null | grep -E "PASS|FAIL"
echo "=== DONE $(date +%H:%M:%S)"
