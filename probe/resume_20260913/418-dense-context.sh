#!/usr/bin/env bash
set -eo pipefail
cd generated
flags=(-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp)
export GLIBC_TUNABLES=glibc.malloc.hugetlb=1 STRASSEN_MAXDEPTH=0
gcc "${flags[@]}" strassen-prime.c gemm-duplex-profile.c -o dense-context -lm -lkblas
echo DENSE_CONTEXT cold_library
numactl -N 1 --preferred=17 ./dense-context 1216 17024 1216 3
echo DENSE_CONTEXT after_library
DENSE_PRIME=1 numactl -N 1 --preferred=17 ./dense-context 1216 17024 1216 3
