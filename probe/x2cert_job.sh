#!/usr/bin/env bash
# x2cert：XONE v2 认证（单节点份×2 提交）——bank+XONE vs 生产 cursor，官方 bench ABBA x8 + gen 全套 + TC2/3 守护
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S) $(git log --oneline -1)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
BIN="trsm_x2_$$"
gcc $FLAGS bench_trsm.c trsm.c -o $BIN -lm -lkblas 2>&1 | grep -E "rror" | head -3
[[ -x $BIN ]] || { echo "BUILD FAILED"; exit 1; }
echo "=== gen（XONE 全组合）"
gcc -O3 -fopenmp -DTRSM_NO_SME_ASM -Dl_trsm=l_trsm_ref -c trsm.c -o probe/refx2_$$.o 2>&1 | grep -E "error" || true
gcc $FLAGS probe/gen_test.c trsm.c probe/refx2_$$.o -o probe/gen_x2_$$ -lm 2>&1 | grep -E "error" || true
X="TRSM_OWNER_CURSOR=0 TRSM_XONE=1"
echo "[gen X owner30]";   env TRSM_OWNER_CURSOR=0 TRSM_XONE=1 TRSM_OWNER=1 timeout 600 numactl -N 1 ./probe/gen_x2_$$ 30 2>&1 | tail -1
echo "[gen X steal20]";   env TRSM_OWNER_CURSOR=0 TRSM_XONE=1 TRSM_OWNER=1 TRSM_STEAL=1 timeout 600 numactl -N 1 ./probe/gen_x2_$$ 20 2>&1 | tail -1
echo "[gen X cfold20]";   env TRSM_OWNER_CURSOR=0 TRSM_XONE=1 TRSM_OWNER=1 TRSM_CFOLD=1 timeout 600 numactl -N 1 ./probe/gen_x2_$$ 20 2>&1 | tail -1
echo "[gen X 5thr20]";    env TRSM_OWNER_CURSOR=0 TRSM_XONE=1 TRSM_OWNER=1 OMP_NUM_THREADS=5 timeout 600 numactl -N 1 ./probe/gen_x2_$$ 20 2>&1 | tail -1
run1(){ local lbl="$1"; shift; echo "--- $lbl"; env "$@" OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./$BIN ${ARGS} 2>&1 | grep -E " x |PASS|FAIL"; }
echo "=== TC1 ABBA x8（prod-cursor vs bank+XONE）"
for r in 1 2 3 4; do
  ARGS="512 19968 5" run1 "tc1_prod_a$r"
  ARGS="512 19968 5" run1 "tc1_xone_a$r" TRSM_OWNER_CURSOR=0 TRSM_XONE=1
  ARGS="512 19968 5" run1 "tc1_xone_b$r" TRSM_OWNER_CURSOR=0 TRSM_XONE=1
  ARGS="512 19968 5" run1 "tc1_prod_b$r"
done
echo "=== TC2/TC3 守护（XONE 不触发）"
for args in "2432 17024 3" "17024 512 3"; do
  ARGS="$args" run1 "tc_${args%% *}_prod"
  ARGS="$args" run1 "tc_${args%% *}_xone" TRSM_OWNER_CURSOR=0 TRSM_XONE=1
done
rm -f $BIN probe/refx2_$$.o probe/gen_x2_$$
echo "=== DONE $(date +%H:%M:%S)"
