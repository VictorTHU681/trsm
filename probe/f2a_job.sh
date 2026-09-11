#!/usr/bin/env bash
# fresh2：装载阶段预取（TRSM_LOADPF）与 Xp 非临时写（TRSM_XPNT）A/B，case2/case3，官方 bench 交错 2 轮 + profile
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
rm -f trsm_test; gcc $FLAGS bench_trsm.c trsm.c -o trsm_test -lm -lkblas 2>&1 | grep -E "rror" | head -5
for i in 1 2; do
  for v in "A=1" "TRSM_LOADPF=2" "TRSM_LOADPF=4" "TRSM_LOADPF=8" "TRSM_XPNT=1" "TRSM_LOADPF=4 TRSM_XPNT=1"; do
    printf "%-30s " "[$v]"; env $v OMP_NUM_THREADS=38 numactl -N 1 ./trsm_test 2432 17024 1 2>&1 | grep -E "PASS|FAIL"
  done
done
for v in "A=1" "TRSM_LOADPF=4" "TRSM_XPNT=1"; do
  printf "%-30s " "[$v]"; env $v OMP_NUM_THREADS=38 numactl -N 1 ./trsm_test 17024 512 1 2>&1 | grep -E "PASS|FAIL"
  printf "%-30s " "[$v]"; env $v OMP_NUM_THREADS=38 numactl -N 1 ./trsm_test 512 19968 1 2>&1 | grep -E "PASS|FAIL"
done
echo "=== profile case2"
for v in "A=1" "TRSM_LOADPF=4" "TRSM_XPNT=1" "TRSM_RESERVE=0"; do echo "--- [$v]"; env $v TRSM_PROFILE=2 OMP_NUM_THREADS=38 numactl -N 1 ./trsm_test 2432 17024 1 2>&1 | grep -E "每线程" | tail -1; done
gcc -O3 -fopenmp -DTRSM_NO_SME_ASM -Dl_trsm=l_trsm_ref -c trsm.c -o probe/ref.o 2>&1 | grep -E "error"
gcc $FLAGS probe/gen_test.c trsm.c probe/ref.o -o probe/gen_test -lm 2>&1 | grep -E "error"
TRSM_LOADPF=4 TRSM_XPNT=1 timeout 600 numactl -N 1 ./probe/gen_test 30 2>&1 | tail -1
echo "=== DONE $(date +%H:%M:%S)"
