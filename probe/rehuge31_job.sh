#!/usr/bin/env bash
# REHUGE@v56 验证：泛化正确性（含 <2MiB skip 路径）+ 官方三用例 REHUGE=1/0 交错 ×3 + 一次性触发确认
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S) $(git log --oneline -1)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
rm -f trsm_test
gcc $FLAGS bench_trsm.c trsm.c -o trsm_test -lm -lkblas 2>&1 | grep -E "rror" | head -5 || true
[ -x ./trsm_test ] || { echo "BUILD FAILED"; exit 1; }
sha256sum trsm.c
echo "=== 一次性触发确认（SMAPS 打印应只出现 1 次/用例）"
TRSM_REHUGE=1 TRSM_SMAPS=1 OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./trsm_test 512 19968 3 2>&1 | grep -cE "trsm rehuge" || true
echo "=== 官方三用例 REHUGE=1/0 交错 ×3"
run1() { local lbl="$1"; shift; echo "--- RUN $lbl"; env "$@" OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./trsm_test ${ARGS} 2>&1 | grep -E "x|PASS|FAIL" | tail -1; }
for r in 1 2 3; do
  for args in "512 19968 5" "2432 17024 3" "17024 512 3"; do
    ARGS="$args" run1 "off_${args%% *}_$r"
    ARGS="$args" run1 "on_${args%% *}_$r" TRSM_REHUGE=1
  done
done
echo "=== 泛化正确性（REHUGE=1 全程开，含小形状 skip 路径）"
gcc -O3 -fopenmp -DTRSM_NO_SME_ASM -Dl_trsm=l_trsm_ref -c trsm.c -o probe/ref.o 2>&1 | grep -E "error" || true
gcc $FLAGS probe/gen_test.c trsm.c probe/ref.o -o probe/gen_test -lm 2>&1 | grep -E "error" || true
echo "[gen REHUGE=1 60]";               TRSM_REHUGE=1 timeout 900 numactl -N 1 ./probe/gen_test 60 2>&1 | tail -1
echo "[gen REHUGE=1 强制owner 30]";     TRSM_REHUGE=1 TRSM_OWNER=1 timeout 600 numactl -N 1 ./probe/gen_test 30 2>&1 | tail -1
echo "[gen REHUGE=1 owner关 20]";       TRSM_REHUGE=1 TRSM_OWNER=0 timeout 600 numactl -N 1 ./probe/gen_test 20 2>&1 | tail -1
echo "[gen REHUGE=1 5线程 20]";         TRSM_REHUGE=1 OMP_NUM_THREADS=5 timeout 600 numactl -N 1 ./probe/gen_test 20 2>&1 | tail -1
echo "[gen 默认(REHUGE=0) 30]";         timeout 600 numactl -N 1 ./probe/gen_test 30 2>&1 | tail -1
echo "=== DONE $(date +%H:%M:%S)"
