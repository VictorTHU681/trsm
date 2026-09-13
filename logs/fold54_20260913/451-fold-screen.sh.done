#!/usr/bin/env bash
set -eo pipefail
cd generated
flags=(-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp)
export GLIBC_TUNABLES=glibc.malloc.hugetlb=1
gcc "${flags[@]}" independent-fold.c trsm-fold.c -o independent-fold -lm
OMP_NUM_THREADS=5 numactl -N 1 --preferred=17 ./independent-fold
gcc "${flags[@]}" bench_trsm.c trsm-fold.c -o fold -lm -lkblas
for round in 1 2 3 4; do
    order=(control off on)
    [[ $((round%2)) == 0 ]] && order=(on off control)
    for mode in "${order[@]}"; do
        binary=fold
        export TRSM_FOLD_OWNER=0
        [[ $mode == control ]] && binary=control
        [[ $mode == on ]] && export TRSM_FOLD_OWNER=1
        echo "FOLD54_FORMAL round=$round mode=$mode m=512 n=19968"
        numactl -N 1 --preferred=17 "./$binary" 512 19968 7
    done
done
for mode in 0 1; do
    echo "FOLD54_PROFILE mode=$mode m=512 n=19968"
    TRSM_FOLD_OWNER=$mode TRSM_PROFILE=3 numactl -N 1 --preferred=17 ./fold 512 19968 5
done
sha256sum trsm-v54.c trsm-fold.c bench_trsm.c
