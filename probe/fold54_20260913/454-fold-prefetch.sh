#!/usr/bin/env bash
set -eo pipefail
cd generated
flags=(-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp)
export GLIBC_TUNABLES=glibc.malloc.hugetlb=1
gcc "${flags[@]}" independent-fold.c trsm-fold-prefetch.c -o independent-fold-prefetch -lm
OMP_NUM_THREADS=5 TRSM_FOLD_KERNEL=3 TRSM_FOLD_PREF=3 numactl -N 1 --preferred=17 ./independent-fold-prefetch
gcc "${flags[@]}" bench_trsm.c trsm-fold-prefetch.c -o fold-prefetch -lm -lkblas
for round in 1 2 3; do
    order=(control old off 0 1 2 3)
    [[ $((round%2)) == 0 ]] && order=(3 2 1 0 off old control)
    for mode in "${order[@]}"; do
        binary=fold-prefetch
        export TRSM_FOLD_OWNER=1 TRSM_FOLD_KERNEL=3 TRSM_FOLD_PREF=0
        [[ $mode == control ]] && binary=control
        [[ $mode == old ]] && binary=fold-kernels
        [[ $mode == off ]] && export TRSM_FOLD_OWNER=0
        [[ $mode != control && $mode != old && $mode != off ]] && export TRSM_FOLD_PREF=$mode
        echo "FOLD54_PREFETCH round=$round mode=$mode m=512 n=19968"
        numactl -N 1 --preferred=17 "./$binary" 512 19968 7
    done
done
for mode in 0 1 2 3; do
    echo "FOLD54_PROFILE mode=$mode m=512 n=19968"
    TRSM_FOLD_OWNER=1 TRSM_FOLD_PREF=$mode TRSM_PROFILE=3 numactl -N 1 --preferred=17 ./fold-prefetch 512 19968 5
done
sha256sum trsm-fold-prefetch.c
