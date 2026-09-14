#!/usr/bin/env bash
# xp95：2a 白发修复（xpd 定向）+ TC3 池税（BLAS_NUM_THREADS=38）联合筛查
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S) $(git log --oneline -1)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
git show e270033:trsm.c > /tmp/trsm_v64b_$$.c
rm -f trsm_xp95 trsm_v64b
gcc $FLAGS bench_trsm.c trsm.c -o trsm_xp95 -lm -lkblas 2>&1 | grep -E "rror" | head -3
gcc $FLAGS bench_trsm.c /tmp/trsm_v64b_$$.c -o trsm_v64b -lm -lkblas 2>&1 | grep -E "rror" | head -3
[[ -x trsm_xp95 && -x trsm_v64b ]] || { echo "BUILD FAILED"; exit 1; }
echo "=== 单测"
gcc $FLAGS probe/xktest.c trsm.c -o probe/xkt95 -lm 2>&1 | grep -E "rror" | head -3
[[ -x probe/xkt95 ]] && numactl -N 1 ./probe/xkt95 2>&1 | tail -2
echo "=== gen"
gcc -O3 -fopenmp -DTRSM_NO_SME_ASM -Dl_trsm=l_trsm_ref -c trsm.c -o probe/ref95.o 2>&1 | grep -E "error" || true
gcc $FLAGS probe/gen_test.c trsm.c probe/ref95.o -o probe/gen95 -lm 2>&1 | grep -E "error" || true
echo "[owner20]"; TRSM_OWNER=1 timeout 600 numactl -N 1 ./probe/gen95 20 2>&1 | tail -1
echo "[steal20]"; TRSM_OWNER=1 TRSM_STEAL=1 timeout 600 numactl -N 1 ./probe/gen95 20 2>&1 | tail -1
run1(){ local lbl="$1" bin="$2"; echo "--- $lbl"; OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./$bin 512 19968 4 2>&1 | grep -E " x |PASS|FAIL"; }
echo "=== TC1 ABBA x6"
for r in 1 2 3 4 5 6; do
  run1 "new_a$r" trsm_xp95
  run1 "old_a$r" trsm_v64b
  run1 "old_b$r" trsm_v64b
  run1 "new_b$r" trsm_xp95
done
echo "=== TC3 池税（wake 桶 + 时间，同 new 二进制）"
for r in 1 2 3; do
  echo "--- tc3_noblas_$r"
  TRSM_PROFILE=2 OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./trsm_xp95 17024 512 4 2>&1 | grep -E " x |PASS|FAIL|wake|唤醒"
  echo "--- tc3_blas38_$r"
  BLAS_NUM_THREADS=38 TRSM_PROFILE=2 OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./trsm_xp95 17024 512 4 2>&1 | grep -E " x |PASS|FAIL|wake|唤醒"
done
echo "=== TC1/TC2 在 =38 下冒烟"
BLAS_NUM_THREADS=38 OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./trsm_xp95 512 19968 3 2>&1 | grep -E " x |PASS|FAIL"
BLAS_NUM_THREADS=38 OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./trsm_xp95 2432 17024 3 2>&1 | grep -E " x |PASS|FAIL"
rm -f trsm_xp95 trsm_v64b /tmp/trsm_v64b_$$.c probe/ref95.o probe/gen95 probe/xkt95
echo "=== DONE $(date +%H:%M:%S)"
