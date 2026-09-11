#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S) $(git log --oneline -1)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
rm -f trsm_test; gcc $FLAGS bench_trsm.c trsm.c -o trsm_test -lm -lkblas 2>&1 | grep -E "rror" | head -5
CFGS=("TRSM_PFX=0" "TRSM_PFX=1" "TRSM_PFX=1 TRSM_PFX_CH=3" "TRSM_PFX=1 TRSM_PFX_CH=10")
for i in 1 2 3 4 5; do for v in "${CFGS[@]}"; do printf "%-28s " "[$v]"; env $v OMP_NUM_THREADS=38 numactl -N 1 ./trsm_test 512 19968 1 2>&1 | grep -E "PASS|FAIL"; done; done
for v in "TRSM_PFX=0" "TRSM_PFX=1"; do echo "[$v]"; env $v TRSM_PROFILE=2 OMP_NUM_THREADS=38 numactl -N 1 ./trsm_test 512 19968 1 2>&1 | grep -E "每线程平均" | tail -1 | cut -c1-330; done
gcc -O3 -fopenmp -DTRSM_NO_SME_ASM -Dl_trsm=l_trsm_ref -c trsm.c -o probe/ref.o 2>&1 | grep -E "error"
gcc $FLAGS probe/gen_test.c trsm.c probe/ref.o -o probe/gen_test -lm 2>&1 | grep -E "error"
TRSM_OWNER=1 timeout 600 numactl -N 1 ./probe/gen_test 30 2>&1 | grep -E "FAIL|PASS" | head -3
echo "=== DONE $(date +%H:%M:%S)"
