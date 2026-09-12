#!/usr/bin/env bash
set -eo pipefail
cd generated
flags=(-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp)
export GLIBC_TUNABLES=glibc.malloc.hugetlb=1 STRASSEN_MAXDEPTH=0 DENSE_ORDER=0
gcc "${flags[@]}" strassen-check.c gemm-duplex-clear.c -o dense-clear -lm -lkblas
for clear in 0 1 2; do
    echo "DENSE_CLEAR_CHECK clear=$clear"
    DENSE_CLEAR=$clear numactl -N 1 --preferred=17 ./dense-clear 208 160 176 1
done
for round in 1 2; do
    for mb in 128 192 256; do
        for clear in 0 1 2; do
            echo "DENSE_CLEAR round=$round mb=$mb clear=$clear"
            DENSE_MB=$mb DENSE_CLEAR=$clear numactl -N 1 --preferred=17 ./dense-clear 1216 17024 1216 2
        done
    done
done
