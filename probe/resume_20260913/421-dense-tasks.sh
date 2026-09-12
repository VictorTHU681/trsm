#!/usr/bin/env bash
set -eo pipefail
cd generated
flags=(-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp)
export GLIBC_TUNABLES=glibc.malloc.hugetlb=1 STRASSEN_MAXDEPTH=0
gcc "${flags[@]}" strassen-profile.c gemm-duplex-tasks.c -o dense-tasks -lm -lkblas
for threads in 1 19 38; do
    echo "DENSE_TASKS threads=$threads mb=192 order=0"
    OMP_NUM_THREADS=$threads numactl -N 1 --preferred=17 ./dense-tasks 1216 17024 1216 2
done
for round in 1 2; do
    for mb in 64 128 192 256; do
        for order in 0 1; do
            echo "DENSE_TASKS round=$round threads=38 mb=$mb order=$order"
            OMP_NUM_THREADS=38 DENSE_MB=$mb DENSE_ORDER=$order numactl -N 1 --preferred=17 ./dense-tasks 1216 17024 1216 2
        done
    done
done
