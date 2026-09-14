#!/usr/bin/env bash
# dn1：DNTC 筛查（TC2 主场 + TC3 顺带）——同二进制 env A/B x6 + gen
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S) $(git log --oneline -1)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
rm -f trsm_dn1
gcc $FLAGS bench_trsm.c trsm.c -o trsm_dn1 -lm -lkblas 2>&1 | grep -E "rror" | head -3
[[ -x trsm_dn1 ]] || { echo "BUILD FAILED"; exit 1; }
echo "=== gen"
gcc -O3 -fopenmp -DTRSM_NO_SME_ASM -Dl_trsm=l_trsm_ref -c trsm.c -o probe/refn1.o 2>&1 | grep -E "error" || true
gcc $FLAGS probe/gen_test.c trsm.c probe/refn1.o -o probe/genn1 -lm 2>&1 | grep -E "error" || true
echo "[DNTC=1 30]"; TRSM_DNTC=1 timeout 700 numactl -N 1 ./probe/genn1 30 2>&1 | tail -1
run2(){ local lbl="$1" args="$2"; shift 2; echo "--- $lbl"; env "$@" TRSM_PROFILE=2 OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./trsm_dn1 $args 2>&1 | grep -E " x |PASS|FAIL|每线程平均"; }
echo "=== TC2 ABBA x6"
for r in 1 2 3 4 5 6; do
  run2 "t2base_$r" "2432 17024 4"
  run2 "t2dnt_$r"  "2432 17024 4" TRSM_DNTC=1
  run2 "t2dnt_b$r" "2432 17024 4" TRSM_DNTC=1
  run2 "t2base_b$r" "2432 17024 4"
done
echo "=== TC3 ABBA x3"
for r in 1 2 3; do
  run2 "t3base_$r" "17024 512 3"
  run2 "t3dnt_$r"  "17024 512 3" TRSM_DNTC=1
done
echo "=== TC1 守护"
TRSM_DNTC=1 OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./trsm_dn1 512 19968 3 2>&1 | grep -E " x |PASS|FAIL"
rm -f trsm_dn1 probe/refn1.o probe/genn1
echo "=== DONE $(date +%H:%M:%S)"
