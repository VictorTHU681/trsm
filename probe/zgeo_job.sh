#!/usr/bin/env bash
# fresh28 ZGEO 冒烟：编译 → gen_test 泛化（开/关）→ TC1 形状 bench 校验 → 同二进制 A/B 三轮
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S)"
. probe/env.sh >/dev/null 2>&1 || true
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
gcc $FLAGS bench_trsm.c trsm.c -o /tmp/trsm_zgeo -lm -lkblas 2>&1 | grep -E "error" && { echo BUILD_FAIL; exit 1; }
gcc $FLAGS probe/gen_test.c trsm.c probe/ref.o -o /tmp/gen_zgeo -lm 2>&1 | grep -E "error"
export OMP_NUM_THREADS=38 OMP_PROC_BIND=close OMP_PLACES=cores OMP_WAIT_POLICY=active GOMP_SPINCOUNT=infinite
echo "--- gen_test 40 ZGEO=1"
TRSM_ZGEO=1 timeout 900 numactl -N 1 /tmp/gen_zgeo 40 2>&1 | tail -1
echo "--- gen_test 20 ZGEO=1 强制 owner（MB=64）"
TRSM_ZGEO=1 TRSM_MB=64 timeout 600 numactl -N 1 /tmp/gen_zgeo 20 2>&1 | tail -1
echo "--- TC1 形状校验 + A/B（同二进制，交错 3 对）"
for i in 1 2 3; do
    echo "== off $i"; TRSM_ZGEO=0 numactl -N 1 --preferred=17 /tmp/trsm_zgeo 512 19968 1 2>/dev/null | grep -E "512|PASS|FAIL"
    echo "== on  $i"; TRSM_ZGEO=1 numactl -N 1 --preferred=17 /tmp/trsm_zgeo 512 19968 1 2>/dev/null | grep -E "512|PASS|FAIL"
done
echo "--- 分相（ZGEO=1，看 kloop 桶）"
TRSM_ZGEO=1 TRSM_PROFILE=2 numactl -N 1 --preferred=17 /tmp/trsm_zgeo 512 19968 1 2>&1 | grep -E "prof|PASS|FAIL" | head -8
echo "=== DONE $(date +%H:%M:%S)"
