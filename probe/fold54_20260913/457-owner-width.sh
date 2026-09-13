#!/usr/bin/env bash
set -eo pipefail
cd generated
flags=(-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp)
export GLIBC_TUNABLES=glibc.malloc.hugetlb=1
gcc "${flags[@]}" independent-width.c trsm-width.c -o independent-width -lm
OMP_NUM_THREADS=5 numactl -N 1 --preferred=17 ./independent-width
gcc "${flags[@]}" bench_trsm.c trsm-width.c -o width -lm -lkblas
for round in 1 2 3 4 5 6; do
    order=(control off on)
    [[ $((round%2)) == 0 ]] && order=(on off control)
    for mode in "${order[@]}"; do
        binary=width
        export TRSM_WIDTH_OWNER=0
        [[ $mode == control ]] && binary=control
        [[ $mode == on ]] && export TRSM_WIDTH_OWNER=1
        echo "OWNER_WIDTH_FORMAL round=$round mode=$mode m=512 n=19968"
        numactl -N 1 --preferred=17 "./$binary" 512 19968 9
    done
done
for mode in 0 1; do
    echo "OWNER_WIDTH_PROFILE mode=$mode m=512 n=19968"
    TRSM_WIDTH_OWNER=$mode TRSM_PROFILE=3 numactl -N 1 --preferred=17 ./width 512 19968 5
done
sha256sum trsm-width.c
