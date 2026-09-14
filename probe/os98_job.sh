#!/usr/bin/env bash
# os98：OSW 相克消融——osw=1 与各特性两两隔离，找官方 TC1 FAIL 的相互作用项
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S) $(git log --oneline -1)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
rm -f trsm_os98
gcc $FLAGS bench_trsm.c trsm.c -o trsm_os98 -lm -lkblas 2>&1 | grep -E "rror" | head -3
[[ -x trsm_os98 ]] || { echo "BUILD FAILED"; exit 1; }
run1(){ local lbl="$1"; shift; echo "--- $lbl"; env "$@" OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./trsm_os98 512 19968 2 2>&1 | grep -E " x |PASS|FAIL|最大误差" | head -3; }
run1 "base"
run1 "osw"            TRSM_OSW=1
run1 "osw_noxone"     TRSM_OSW=1 TRSM_XONE=0
run1 "osw_nocfold"    TRSM_OSW=1 TRSM_CFOLD=0
run1 "osw_nosteal"    TRSM_OSW=1 TRSM_STEAL=0
run1 "osw_noshapep"   TRSM_OSW=1 TRSM_SHAPEP=0
run1 "osw_noslx"      TRSM_OSW=1 TRSM_SLX=0
run1 "osw_min"        TRSM_OSW=1 TRSM_XONE=0 TRSM_CFOLD=0 TRSM_STEAL=0 TRSM_SHAPEP=0 TRSM_SLX=0
rm -f trsm_os98
echo "=== DONE $(date +%H:%M:%S)"
