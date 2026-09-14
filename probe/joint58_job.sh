#!/usr/bin/env bash
# joint58：v57 生产栈（REHUGE∧LTO∧SHAPEP）联合无回归认证（2026-09-14 双审计指出三因子从未同栈 A/B）
# 设计：同一棵树出两个二进制（LTO 档 = run.sh 首选行 / plain 档 = run.sh 回退行，独立命名防并行作业互删），
#   TC1 四线（FULL / 无LTO / 无REHUGE / 无SHAPEP）×6 轮交错；TC2/TC3 三线（SHAPEP 仅作用 owner/TC1）×3 轮。
#   本地干净节点预期：REHUGE 线 ±0（覆盖率本就高，验无交互回归）；LTO 线复现 +0.5~1.5%；SHAPEP 线复现 +1%。
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S) $(git log --oneline -1)"
. probe/env.sh
BASE="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
rm -f trsm_j58l trsm_j58p
gcc $BASE -mtune=tsv110 -flto bench_trsm.c trsm.c -o trsm_j58l -lm -lkblas 2>probe/j58_lto.err || true
gcc $BASE bench_trsm.c trsm.c -o trsm_j58p -lm -lkblas 2>&1 | grep -E "rror" | head -3 || true
[ -x ./trsm_j58p ] || { echo "BUILD FAILED (plain)"; exit 1; }
if [ -x ./trsm_j58l ]; then echo "[build] LTO OK"; else echo "[build] LTO FAILED:"; head -5 probe/j58_lto.err; exit 1; fi
sha256sum trsm.c trsm_j58l trsm_j58p | cut -c1-72

echo "=== gen 快检（LTO 二进制 + REHUGE 默认开，20 例）"
gcc -O3 -fopenmp -DTRSM_NO_SME_ASM -Dl_trsm=l_trsm_ref -c trsm.c -o probe/ref58.o 2>&1 | grep -E "error" || true
gcc $BASE -mtune=tsv110 -flto probe/gen_test.c trsm.c probe/ref58.o -o probe/gen_test58 -lm 2>/dev/null \
  || gcc $BASE probe/gen_test.c trsm.c probe/ref58.o -o probe/gen_test58 -lm 2>&1 | grep -E "error" || true
timeout 600 numactl -N 1 ./probe/gen_test58 20 2>&1 | tail -1

run1() { local lbl="$1" bin="$2"; shift 2; echo "--- RUN $lbl"; env "$@" OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./$bin ${ARGS} 2>&1 | grep -E "x|PASS|FAIL" | tail -1; }

echo "=== TC1 四线交错 ×6（FULL / noLTO / noREHUGE / noSHAPEP）"
for r in 1 2 3 4 5 6; do
  ARGS="512 19968 5" run1 "tc1_full_$r"   trsm_j58l
  ARGS="512 19968 5" run1 "tc1_nolto_$r"  trsm_j58p
  ARGS="512 19968 5" run1 "tc1_norh_$r"   trsm_j58l TRSM_REHUGE=0
  ARGS="512 19968 5" run1 "tc1_nosp_$r"   trsm_j58l TRSM_SHAPEP=0
done

echo "=== TC2/TC3 三线交错 ×3（FULL / noLTO / noREHUGE）"
for r in 1 2 3; do
  for args in "2432 17024 3" "17024 512 3"; do
    ARGS="$args" run1 "tc_${args%% *}_full_$r"  trsm_j58l
    ARGS="$args" run1 "tc_${args%% *}_nolto_$r" trsm_j58p
    ARGS="$args" run1 "tc_${args%% *}_norh_$r"  trsm_j58l TRSM_REHUGE=0
  done
done
rm -f probe/ref58.o probe/gen_test58
echo "=== DONE $(date +%H:%M:%S)"
