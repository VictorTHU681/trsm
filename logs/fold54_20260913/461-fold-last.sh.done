#!/usr/bin/env bash
set -eo pipefail
cd generated
flags=(-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp)
export GLIBC_TUNABLES=glibc.malloc.hugetlb=1
gcc "${flags[@]}" bench_trsm.c trsm-fold-last.c -o fold-last -lm -lkblas
for round in 1 2 3 4; do
    order=(control off fold lastnt)
    [[ $((round%2)) == 0 ]] && order=(lastnt fold off control)
    for mode in "${order[@]}"; do
        binary=fold-last
        export TRSM_FOLD_OWNER=0 TRSM_FOLD_KERNEL=3 TRSM_FOLD_LASTNT=0 TRSM_FOLD_NOPACK=1 TRSM_FOLD_PREF=0
        [[ $mode == control ]] && binary=control
        [[ $mode == fold || $mode == lastnt ]] && export TRSM_FOLD_OWNER=1
        [[ $mode == lastnt ]] && export TRSM_FOLD_LASTNT=1
        echo "FOLD_LAST_FORMAL round=$round mode=$mode m=512 n=19968"
        numactl -N 1 --preferred=17 "./$binary" 512 19968 9
    done
done
for mode in 0 1; do
    echo "FOLD_LAST_PROFILE mode=$mode m=512 n=19968"
    TRSM_FOLD_OWNER=1 TRSM_FOLD_LASTNT=$mode TRSM_PROFILE=3 numactl -N 1 --preferred=17 ./fold-last 512 19968 5
done
sha256sum trsm-fold-last.c
