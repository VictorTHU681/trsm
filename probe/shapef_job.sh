#!/usr/bin/env bash
# SHAPEP 正式认证：同一二进制 TRSM_SHAPEP=0/100 交错 ×6（TC1 官方 bench ×7 计时）+ TC2/TC3 守护 + 泛化门禁
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S) $(git log --oneline -1)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
rm -f trsm_test
gcc $FLAGS bench_trsm.c trsm.c -o trsm_test -lm -lkblas 2>&1 | grep -E "rror" | head -5 || true
[ -x ./trsm_test ] || { echo "BUILD FAILED"; exit 1; }
sha256sum trsm.c trsm_test
run1() { local lbl="$1"; shift; echo "--- RUN $lbl"; env "$@" OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./trsm_test ${ARGS:-512 19968 7} 2>&1 | grep -E "x|PASS|FAIL" | tail -1; }
echo "=== TC2/TC3 守护（SHAPEP 只影响 owner 路径，确认无扰动）"
ARGS="2432 17024 3" run1 tc2_on  TRSM_SHAPEP=100
ARGS="17024 512 3" run1 tc3_on  TRSM_SHAPEP=100
echo "=== TC1 正式交错 ×6"
for r in 1 2 3 4 5 6; do
  run1 "off_$r" TRSM_SHAPEP=0
  run1 "on_$r"  TRSM_SHAPEP=100
done
echo "=== 泛化门禁"
gcc -O3 -fopenmp -DTRSM_NO_SME_ASM -Dl_trsm=l_trsm_ref -c trsm.c -o probe/ref.o 2>&1 | grep -E "error" || true
gcc $FLAGS probe/gen_test.c trsm.c probe/ref.o -o probe/gen_test -lm 2>&1 | grep -E "error" || true
echo "[gen 默认(SHAPEP=100) 60]";        timeout 900 numactl -N 1 ./probe/gen_test 60 2>&1 | tail -1
echo "[gen SHAPEP=0 30]";                TRSM_SHAPEP=0 timeout 600 numactl -N 1 ./probe/gen_test 30 2>&1 | tail -1
echo "[gen 强制 owner + SHAPEP 30]";     TRSM_OWNER=1 timeout 600 numactl -N 1 ./probe/gen_test 30 2>&1 | tail -1
echo "[gen owner 关 20]";                TRSM_OWNER=0 timeout 600 numactl -N 1 ./probe/gen_test 20 2>&1 | tail -1
echo "[gen 5 线程 + SHAPEP 20]";         OMP_NUM_THREADS=5 timeout 600 numactl -N 1 ./probe/gen_test 20 2>&1 | tail -1
echo "[gen SWHOST MB64 20]";             TRSM_SWHOST=1 TRSM_MB=64 timeout 600 numactl -N 1 ./probe/gen_test 20 2>&1 | tail -1
echo "=== DONE $(date +%H:%M:%S)"
