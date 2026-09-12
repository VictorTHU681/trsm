#!/usr/bin/env bash
set -eo pipefail
cd generated
flags=(-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp)
export GLIBC_TUNABLES=glibc.malloc.hugetlb=1
for mb in 64 128 192 256; do
    gcc "${flags[@]}" bench_trsm.c trsm-special-mb$mb.c -o special-mb$mb -lm -lkblas
done
for round in 1 2; do
    for shape in '2432 17024 2' '17024 512 4'; do
        read -r m n special <<<"$shape"
        order=(64 128 192 256)
        [[ $round == 2 ]] && order=(256 192 128 64)
        for mb in "${order[@]}"; do
            echo "SPECIAL_MB round=$round m=$m n=$n mb=$mb special=$special"
            TRSM_MB=$mb TRSM_EXPECT_SPECIAL=$special numactl -N 1 --preferred=17 ./special-mb$mb "$m" "$n" 5
        done
    done
done
for shape in '2432 17024 2' '17024 512 4'; do
    read -r m n special <<<"$shape"
    for mb in 64 128 192 256; do
        echo "SPECIAL_MB_PROFILE m=$m n=$n mb=$mb special=$special"
        TRSM_MB=$mb TRSM_EXPECT_SPECIAL=$special TRSM_PROFILE=2 numactl -N 1 --preferred=17 ./special-mb$mb "$m" "$n" 3
    done
done
