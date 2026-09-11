#!/usr/bin/env bash
# fresh：GEMM 段内循环顺序（TRSM_TORDER=1：列块外层、条内层）对宽 nc 的影响（通用路径 SPECIAL=0）
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
rm -f fastbench; gcc $FLAGS probe/fastbench.c trsm.c -o fastbench -lm 2>&1 | grep -E "rror" | head -5; [[ -x fastbench ]] || { echo COMPILE_FAIL; exit 1; }
run() { local mn="$1"; shift; echo "--- [$mn] [$*]"; env "$@" OMP_NUM_THREADS=38 numactl -N 1 ./fastbench -r 3 $mn 2>&1 | grep -vE "^  rep"; }
echo "=== gen_test torder"
gcc -O3 -fopenmp -DTRSM_NO_SME_ASM -Dl_trsm=l_trsm_ref -c trsm.c -o probe/ref.o 2>&1 | grep -E "error"
gcc $FLAGS probe/gen_test.c trsm.c probe/ref.o -o probe/gen_test -lm 2>&1 | grep -E "error"
TRSM_TORDER=1 TRSM_OWNER=0 TRSM_DIRECT=0 TRSM_SPECIAL=0 timeout 600 numactl -N 1 ./probe/gen_test 30 2>&1 | tail -1
TRSM_TORDER=1 TRSM_OWNER=1 timeout 600 numactl -N 1 ./probe/gen_test 20 2>&1 | tail -1
TRSM_TORDER=1 TRSM_SPECIAL=0 TRSM_MB=192 TRSM_NC=256 timeout 600 numactl -N 1 ./probe/gen_test 20 2>&1 | tail -1
C3="17024 512"; C2="2432 17024"
for cfg in "192 128" "192 256" "192 512" "128 256" "128 512" "96 512" "256 256"; do
  set -- $cfg
  run "$C3" TRSM_SPECIAL=0 TRSM_MB=$1 TRSM_NC=$2 TRSM_TORDER=0
  run "$C3" TRSM_SPECIAL=0 TRSM_MB=$1 TRSM_NC=$2 TRSM_TORDER=1
done
for cfg in "256 128" "192 256" "256 256" "128 256"; do
  set -- $cfg
  run "$C2" TRSM_SPECIAL=0 TRSM_MB=$1 TRSM_NC=$2 TRSM_TORDER=0
  run "$C2" TRSM_SPECIAL=0 TRSM_MB=$1 TRSM_NC=$2 TRSM_TORDER=1
done
run "512 19968" TRSM_TORDER=1
echo "=== DONE $(date +%H:%M:%S)"
