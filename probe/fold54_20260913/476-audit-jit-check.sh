#!/usr/bin/env bash
set -eo pipefail
cd generated
ulimit -c 0
export GLIBC_TUNABLES=glibc.malloc.hugetlb=1
export OMP_NUM_THREADS=5 TRSM_OWNER=0 TRSM_DIRECT=0 TRSM_MB=192
export TRSM_JIT_MB=0 TRSM_PPMODE=0 TRSM_JIT=1 TRSM_FUSED_DIAG=2
# This counter instrument tracks prepacked strips, not JIT strips.
# Preserve its failure visibly, then validate the uninstrumented solver.
set +e
numactl -N 1 --preferred=17 stdbuf -oL ./gen-fused 10
status=$?
set -e
echo "EXPECTED_PREPACK_AUDIT_EXIT=$status"
[[ $status == 134 ]]
gcc -O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp gen-strict.c fused-final.o reference.o -o gen-fused-jit-check -lm
numactl -N 1 --preferred=17 ./gen-fused-jit-check 30
echo AUDIT_JIT_CHECK_DONE
