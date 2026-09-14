#!/usr/bin/env bash
# x3：XONE v3 双线槽 vs v63 基线（dca36c8）——TC1 ABBA x6 + gen + TC2 冒烟
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S) $(git log --oneline -1)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
git show dca36c8:trsm.c > /tmp/trsm_v63_$$.c
rm -f trsm_v63b trsm_v3b
gcc $FLAGS bench_trsm.c /tmp/trsm_v63_$$.c -o trsm_v63b -lm -lkblas 2>&1 | grep -E "rror" | head -3
gcc $FLAGS bench_trsm.c trsm.c -o trsm_v3b -lm -lkblas 2>&1 | grep -E "rror" | head -3
[[ -x trsm_v63b && -x trsm_v3b ]] || { echo "BUILD FAILED"; exit 1; }
echo "=== gen（v3 全组合）"
gcc -O3 -fopenmp -DTRSM_NO_SME_ASM -Dl_trsm=l_trsm_ref -c trsm.c -o probe/ref3_$$.o 2>&1 | grep -E "error" || true
gcc $FLAGS probe/gen_test.c trsm.c probe/ref3_$$.o -o probe/gen3_$$ -lm 2>&1 | grep -E "error" || true
echo "[gen 默认30]"; timeout 700 numactl -N 1 ./probe/gen3_$$ 30 2>&1 | tail -1
echo "[gen owner+steal+cfold 20]"; TRSM_OWNER=1 timeout 600 numactl -N 1 ./probe/gen3_$$ 20 2>&1 | tail -1
run1(){ local lbl="$1" bin="$2"; shift 2; echo "--- $lbl"; env "$@" OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./$bin 512 19968 5 2>&1 | grep -E " x |PASS|FAIL"; }
echo "=== TC1 ABBA x6"
for r in 1 2 3 4 5 6; do
  run1 "v63_a$r" trsm_v63b
  run1 "v3_a$r"  trsm_v3b
  run1 "v3_b$r"  trsm_v3b
  run1 "v63_b$r" trsm_v63b
done
echo "=== TC2 冒烟"
OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./trsm_v3b 2432 17024 3 2>&1 | grep -E " x |PASS|FAIL"
rm -f trsm_v63b trsm_v3b /tmp/trsm_v63_$$.c probe/ref3_$$.o probe/gen3_$$
echo "=== DONE $(date +%H:%M:%S)"
