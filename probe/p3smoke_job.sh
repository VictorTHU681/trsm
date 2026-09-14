#!/usr/bin/env bash
# p3smoke：env_int 快照表替换 getenv 后的解析正确性冒烟（PROFILE/DWPF/OWNER/CFOLD 等开关经快照路径读取）
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S) $(git log --oneline -1)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
rm -f trsm_p3s; gcc $FLAGS bench_trsm.c trsm.c -o trsm_p3s -lm -lkblas 2>&1 | grep -E "rror" | head -3
[[ -x trsm_p3s ]] || { echo "BUILD FAILED"; exit 1; }
echo "=== 三用例 PASS + PROFILE 解析确认（应出现 prof 行）"
TRSM_PROFILE=1 OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./trsm_p3s 512 19968 3 2>&1 | grep -E " x |PASS|FAIL|parallel" | head -3
OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./trsm_p3s 2432 17024 3 2>&1 | grep -E " x |PASS|FAIL"
OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./trsm_p3s 17024 512 3 2>&1 | grep -E " x |PASS|FAIL"
echo "=== 开关解析确认（CFOLD=0 应回到 13 线基线；REHUGE=0 应无 rehuge 行为）"
TRSM_CFOLD=0 OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./trsm_p3s 512 19968 3 2>&1 | grep -E " x |PASS|FAIL"
echo "=== gen 正确性"
gcc -O3 -fopenmp -DTRSM_NO_SME_ASM -Dl_trsm=l_trsm_ref -c trsm.c -o probe/ref3.o 2>&1 | grep -E "error" || true
gcc $FLAGS probe/gen_test.c trsm.c probe/ref3.o -o probe/gen_test3 -lm 2>&1 | grep -E "error" || true
timeout 600 numactl -N 1 ./probe/gen_test3 25 2>&1 | tail -1
TRSM_OWNER=1 TRSM_CFOLD=1 timeout 600 numactl -N 1 ./probe/gen_test3 20 2>&1 | tail -1
rm -f trsm_p3s probe/ref3.o probe/gen_test3
echo "=== DONE $(date +%H:%M:%S)"
