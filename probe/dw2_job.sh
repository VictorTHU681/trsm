#!/usr/bin/env bash
# dw2：DWPF2 筛查（对角窗 B-next 密度 x2，DIAGPULL 阶段 A）——TC1 ABBA x6 + gen + kloop0 桶
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S) $(git log --oneline -1)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
rm -f trsm_dw2
gcc $FLAGS bench_trsm.c trsm.c -o trsm_dw2 -lm -lkblas 2>&1 | grep -E "rror" | head -3
[[ -x trsm_dw2 ]] || { echo "BUILD FAILED"; exit 1; }
echo "=== gen"
gcc -O3 -fopenmp -DTRSM_NO_SME_ASM -Dl_trsm=l_trsm_ref -c trsm.c -o probe/refw2.o 2>&1 | grep -E "error" || true
gcc $FLAGS probe/gen_test.c trsm.c probe/refw2.o -o probe/genw2 -lm 2>&1 | grep -E "error" || true
echo "[owner+steal DWPF2 25]"; TRSM_DWPF2=1 TRSM_OWNER=1 TRSM_STEAL=1 timeout 700 numactl -N 1 ./probe/genw2 25 2>&1 | tail -1
run1(){ local lbl="$1"; shift; echo "--- $lbl"; env "$@" TRSM_PROFILE=2 OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./trsm_dw2 512 19968 4 2>&1 | grep -E " x |PASS|FAIL|每线程平均"; }
echo "=== TC1 ABBA x6"
for r in 1 2 3 4 5 6; do
  run1 "base_$r"
  run1 "dw2_$r"  TRSM_DWPF2=1
  run1 "dw2_b$r" TRSM_DWPF2=1
  run1 "base_b$r"
done
echo "=== oswrep 测度零形状（DWPF2=1）"
for a in "3 128 768" "3 192 960" "8 512 2496"; do
  set -- $a; thr=$1; m=$2; n=$3
  printf "[%s %sx%s] " "$thr" "$m" "$n"
  TRSM_DWPF2=1 TRSM_OWNER=1 OMP_NUM_THREADS=$thr numactl -N 1 $MEMOPT ./trsm_dw2 $m $n 1 2>&1 | awk '/PASS|FAIL/{print $NF}' | tail -1
done
rm -f trsm_dw2 probe/refw2.o probe/genw2
echo "=== DONE $(date +%H:%M:%S)"
