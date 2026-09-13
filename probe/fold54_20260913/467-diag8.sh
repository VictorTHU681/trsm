#!/usr/bin/env bash
set -eo pipefail
cd generated
flags=(-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp)
export GLIBC_TUNABLES=glibc.malloc.hugetlb=1
gcc "${flags[@]}" -c trsm-diag8.c -o diag8.o
gcc "${flags[@]}" independent-diag8.c diag8.o -o independent-diag8 -lm
OMP_NUM_THREADS=5 numactl -N 1 --preferred=17 ./independent-diag8
gcc "${flags[@]}" bench_trsm.c diag8.o -o diag8 -lm -lkblas
for round in 1 2 3 4 5 6; do
    order=(control off on)
    [[ $((round%2)) == 0 ]] && order=(on off control)
    for mode in "${order[@]}"; do
        binary=diag8
        export TRSM_DIAG8=0
        [[ $mode == control ]] && binary=control
        [[ $mode == on ]] && export TRSM_DIAG8=1
        echo "DIAG8_FORMAL round=$round mode=$mode m=512 n=19968"
        numactl -N 1 --preferred=17 "./$binary" 512 19968 13
    done
done
for mode in 0 1; do
    echo "DIAG8_PROFILE mode=$mode m=512 n=19968"
    TRSM_DIAG8=$mode TRSM_PROFILE=3 numactl -N 1 --preferred=17 ./diag8 512 19968 5
done
sha256sum trsm-diag8.c
