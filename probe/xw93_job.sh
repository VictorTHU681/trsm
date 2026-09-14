#!/usr/bin/env bash
# xw93：终选——base / all3(xpw+dw0+dpf0) / simp(pch+dw0+dpf0，无 XPW) 三方 ABBA x4
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S) $(git log --oneline -1)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
BIN=trsm_xw93_$$
gcc $FLAGS bench_trsm.c trsm.c -o $BIN -lm -lkblas 2>&1 | grep -E "rror" | head -3
[[ -x $BIN ]] || { echo "BUILD FAILED"; exit 1; }
run2(){ local lbl="$1"; shift; echo "--- $lbl"; env "$@" OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./$BIN 2432 17024 4 2>&1 | grep -E " x |PASS|FAIL"; }
for r in 1 2 3 4; do
  run2 "base_a$r"
  run2 "all3_a$r" TRSM_XPW=1 TRSM_DWPF=0 TRSM_DPF=0
  run2 "simp_a$r" TRSM_PCHAIN=1 TRSM_DWPF=0 TRSM_DPF=0
  run2 "simp_b$r" TRSM_PCHAIN=1 TRSM_DWPF=0 TRSM_DPF=0
  run2 "all3_b$r" TRSM_XPW=1 TRSM_DWPF=0 TRSM_DPF=0
  run2 "base_b$r"
done
rm -f $BIN
echo "=== DONE $(date +%H:%M:%S)"
