#!/usr/bin/env bash
set -eo pipefail
cd generated
flags=(-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp)
export GLIBC_TUNABLES=glibc.malloc.hugetlb=1 STRASSEN_MAXDEPTH=0
gcc "${flags[@]}" strassen-profile.c gemm-duplex-pmu.c -o dense-pmu -lm -lkblas
for stride in 0 1; do
    echo "DENSE_PLACEMENT stride=$stride"
    DENSE_CSTRIDE=$stride numactl -N 1 --preferred=17 ./dense-pmu 1216 17024 1216 3
done
