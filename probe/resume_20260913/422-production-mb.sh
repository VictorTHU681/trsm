#!/usr/bin/env bash
set -eo pipefail
cd generated
flags=(-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp)
export GLIBC_TUNABLES=glibc.malloc.hugetlb=1
gcc "${flags[@]}" bench_trsm.c trsm-control.c -o production-mb -lm -lkblas
for round in 1 2; do
    for shape in '2432 17024' '17024 512'; do
        read -r m n <<<"$shape"
        order=(64 128 192 256)
        [[ $round == 2 ]] && order=(256 192 128 64)
        for mb in "${order[@]}"; do
            echo "PRODUCTION_MB round=$round m=$m n=$n mb=$mb"
            TRSM_MB=$mb numactl -N 1 --preferred=17 ./production-mb "$m" "$n" 5
        done
    done
done
for shape in '2432 17024' '17024 512'; do
    read -r m n <<<"$shape"
    for mb in 128 192 256; do
        echo "PRODUCTION_MB_PROFILE m=$m n=$n mb=$mb"
        TRSM_MB=$mb TRSM_PROFILE=2 numactl -N 1 --preferred=17 ./production-mb "$m" "$n" 3
    done
done
