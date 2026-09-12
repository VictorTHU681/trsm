#!/usr/bin/env bash
# 纯净形态验证：launch 侧 --preferred + glibc 大页 tunable 能否替代算子内 X 迁移/REHUGE
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S) $(git log --oneline -1)"
. probe/env.sh
getconf GNU_LIBC_VERSION
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
rm -f trsm_test; gcc $FLAGS bench_trsm.c trsm.c -o trsm_test -lm -lkblas 2>&1 | grep -E "rror" | head -5
run() { local tag="$1"; shift; OMP_NUM_THREADS=38 "$@" ./trsm_test $M $N 1 2>&1 | grep -E "PASS|FAIL" | sed "s/^/[$tag] /"; }
for round in 1 2; do
  for MN in "512 19968" "2432 17024" "17024 512"; do
    read -r M N <<<"$MN"
    # A: 现状（run.sh 口径：--preferred + 算子内回退全开）
    run A numactl -N 1 $MEMOPT
    # B: 纯净（--preferred + 算子不碰调用方缓冲）
    TRSM_XNODE=-1 TRSM_REHUGE=0 run B numactl -N 1 $MEMOPT
    # C: 纯净 + glibc 大页 tunable
    GLIBC_TUNABLES=glibc.malloc.hugetlb=1 TRSM_XNODE=-1 TRSM_REHUGE=0 run C numactl -N 1 $MEMOPT
  done
done
echo "--- THP 覆盖诊断（TC1，X 所在 VMA）"
TRSM_SMAPS=1 TRSM_XNODE=-1 TRSM_REHUGE=0 OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./trsm_test 512 19968 1 2>&1 | grep -E "smaps|PASS" | head -6
GLIBC_TUNABLES=glibc.malloc.hugetlb=1 TRSM_SMAPS=1 TRSM_XNODE=-1 TRSM_REHUGE=0 OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./trsm_test 512 19968 1 2>&1 | grep -E "smaps|PASS" | head -6
echo "=== DONE $(date +%H:%M:%S)"
