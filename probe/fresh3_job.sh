#!/usr/bin/env bash
# fresh：对角条 ldnt1d（TRSM_DNT）对 case2/case3 的影响 + 分相 profile（看 load 是否被对角阶段的预取覆盖）
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
rm -f fastbench; gcc $FLAGS probe/fastbench.c trsm.c -o fastbench -lm 2>&1 | grep -E "rror" | head -5; [[ -x fastbench ]] || { echo COMPILE_FAIL; exit 1; }
run() { local mn="$1"; shift; echo "--- [$mn] [$*]"; env "$@" OMP_NUM_THREADS=38 numactl -N 1 ./fastbench -r 4 $mn 2>&1 | grep -vE "^  rep"; }
prof() { local mn="$1"; shift; echo "--- prof [$mn] [$*]"; env "$@" TRSM_PROFILE=2 OMP_NUM_THREADS=38 numactl -N 1 ./fastbench -r 1 $mn 2>&1 | grep -E "每线程|min " | tail -2; }
for v in "TRSM_DNT=0" "TRSM_DNT=1"; do
  run "2432 17024" $v
  run "17024 512" $v
  prof "2432 17024" $v
  prof "17024 512" $v
done
run "2432 17024" TRSM_DNT=1 TRSM_RESERVE=0
run "2432 17024" TRSM_DNT=0 TRSM_RESERVE=0
echo "=== gen_test（DNT=1 走 special/通用）"
gcc -O3 -fopenmp -DTRSM_NO_SME_ASM -Dl_trsm=l_trsm_ref -c trsm.c -o probe/ref.o 2>&1 | grep -E "error"
gcc $FLAGS probe/gen_test.c trsm.c probe/ref.o -o probe/gen_test -lm 2>&1 | grep -E "error"
TRSM_DNT=1 timeout 600 numactl -N 1 ./probe/gen_test 40 2>&1 | tail -1
TRSM_DNT=1 TRSM_OWNER=0 TRSM_DIRECT=0 timeout 600 numactl -N 1 ./probe/gen_test 30 2>&1 | tail -1
echo "=== DONE $(date +%H:%M:%S)"
