#!/usr/bin/env bash
# fresh：首段融合装载（TRSM_FUSE）正确性 + A/B
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
echo "=== 泛化正确性"
gcc -O3 -fopenmp -DTRSM_NO_SME_ASM -Dl_trsm=l_trsm_ref -c trsm.c -o probe/ref.o 2>&1 | grep -E "error"
gcc $FLAGS probe/gen_test.c trsm.c probe/ref.o -o probe/gen_test -lm 2>&1 | grep -E "error"
timeout 600 numactl -N 1 ./probe/gen_test 60 2>&1 | tail -1
TRSM_OWNER=1 timeout 600 numactl -N 1 ./probe/gen_test 40 2>&1 | tail -1
TRSM_OWNER=0 TRSM_DIRECT=0 TRSM_SPECIAL=0 timeout 600 numactl -N 1 ./probe/gen_test 40 2>&1 | tail -1
TRSM_OWNER=0 TRSM_DIRECT=0 TRSM_SPECIAL=0 TRSM_MB=128 TRSM_NC=128 TRSM_KC=128 timeout 600 numactl -N 1 ./probe/gen_test 20 2>&1 | tail -1
TRSM_OWNER=0 TRSM_DIRECT=0 TRSM_SPECIAL=0 TRSM_MB=64 TRSM_NC=64 TRSM_KC=32 timeout 600 numactl -N 1 ./probe/gen_test 20 2>&1 | tail -1
TRSM_FUSE=0 timeout 600 numactl -N 1 ./probe/gen_test 20 2>&1 | tail -1
echo "=== A/B（fastbench -r 5）"
rm -f fastbench; gcc $FLAGS probe/fastbench.c trsm.c -o fastbench -lm 2>&1 | grep -E "rror" | head -5; [[ -x fastbench ]] || { echo COMPILE_FAIL; exit 1; }
run() { local mn="$1"; shift; echo "--- [$mn] [$*]"; env "$@" OMP_NUM_THREADS=38 numactl -N 1 ./fastbench -r 5 $mn 2>&1 | grep -vE "^  rep"; }
run "512 19968" A=1
run "512 19968" TRSM_FUSE=0
run "512 19968" A=2
run "512 19968" TRSM_FUSE=0
run "2432 17024" TRSM_SPECIAL=0
run "2432 17024" TRSM_SPECIAL=0 TRSM_FUSE=0
run "2432 17024" TRSM_SPECIAL=0 TRSM_XNODE=-2
run "2432 17024" TRSM_SPECIAL=0 TRSM_XNODE=-2 TRSM_FUSE=0
echo "=== TC1 分相 profile（官方 bench）"
rm -f trsm_test; gcc $FLAGS bench_trsm.c trsm.c -o trsm_test -lm -lkblas 2>&1 | grep -E "rror" | head -5
prof1() { echo "--- [$*]"; env "$@" TRSM_PROFILE=2 OMP_NUM_THREADS=38 numactl -N 1 ./trsm_test 512 19968 1 2>&1 | grep -E "PASS|FAIL|prof\]" | tail -4; }
prof1 A=1
prof1 TRSM_FUSE=0
echo "=== 官方 bench 三用例 ×2"
for r in 1 2; do for args in "512 19968 1" "2432 17024 1" "17024 512 1"; do OMP_NUM_THREADS=38 numactl -N 1 ./trsm_test $args 2>&1 | grep -E "PASS|FAIL"; done; done
echo "=== DONE $(date +%H:%M:%S)"
