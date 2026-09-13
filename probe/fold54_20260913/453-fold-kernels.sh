#!/usr/bin/env bash
set -eo pipefail
cd generated
flags=(-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp)
export GLIBC_TUNABLES=glibc.malloc.hugetlb=1
gcc "${flags[@]}" fold-boundary.c -o fold-boundary -lm
OMP_NUM_THREADS=1 numactl -N 1 --preferred=17 ./fold-boundary
gcc "${flags[@]}" bench_trsm.c trsm-fold-kernels.c -o fold-kernels -lm -lkblas
for round in 1 2 3; do
    order=(control off 0 1 2 3 4 5)
    [[ $((round%2)) == 0 ]] && order=(5 4 3 2 1 0 off control)
    for mode in "${order[@]}"; do
        binary=fold-kernels
        export TRSM_FOLD_OWNER=1 TRSM_FOLD_KERNEL=0
        [[ $mode == control ]] && binary=control
        [[ $mode == off ]] && export TRSM_FOLD_OWNER=0
        [[ $mode != control && $mode != off ]] && export TRSM_FOLD_KERNEL=$mode
        echo "FOLD54_KERNEL round=$round mode=$mode m=512 n=19968"
        numactl -N 1 --preferred=17 "./$binary" 512 19968 7
    done
done
for mode in 0 1 2 3 4 5; do
    echo "FOLD54_PROFILE mode=$mode m=512 n=19968"
    TRSM_FOLD_OWNER=1 TRSM_FOLD_KERNEL=$mode TRSM_PROFILE=3 numactl -N 1 --preferred=17 ./fold-kernels 512 19968 5
done
sha256sum trsm-fold-kernels.c fold-boundary.c
