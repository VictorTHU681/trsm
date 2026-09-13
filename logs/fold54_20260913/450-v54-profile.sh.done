#!/usr/bin/env bash
set -eo pipefail
cd generated
flags=(-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp)
export GLIBC_TUNABLES=glibc.malloc.hugetlb=1
gcc "${flags[@]}" bench_trsm.c trsm-v54.c -o control -lm -lkblas
for round in 1 2; do
    for shape in '512 19968' '2432 17024' '17024 512'; do
        echo "FOLD54_BASE round=$round shape=$shape"
        numactl -N 1 --preferred=17 ./control $shape 7
    done
done
for shape in '512 19968' '2432 17024' '17024 512'; do
    echo "FOLD54_PROFILE mode=control shape=$shape"
    TRSM_PROFILE=3 numactl -N 1 --preferred=17 ./control $shape 5
done
