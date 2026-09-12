#!/usr/bin/env bash
set -eo pipefail
cd generated
flags=(-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp)
export GLIBC_TUNABLES=glibc.malloc.hugetlb=1 STRASSEN_MAXDEPTH=0
gcc "${flags[@]}" strassen-profile.c gemm-duplex-layout.c -o dense-layout -lm -lkblas
numactl -N 1 --preferred=17 ./dense-layout 64 96 80 1
numactl -N 1 --preferred=17 ./dense-layout 1216 17024 1216 3
numactl -N 1 --preferred=17 ./dense-layout 8512 512 8512 3
