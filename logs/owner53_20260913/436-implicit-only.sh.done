#!/usr/bin/env bash
set -eo pipefail
cd generated
flags=(-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp)
export GLIBC_TUNABLES=glibc.malloc.hugetlb=1
gcc "${flags[@]}" bench_trsm.c trsm-implicit-only.c -o implicit-only -lm -lkblas
gcc "${flags[@]}" independent-owner53.c trsm-implicit-only.c -o independent-implicit-only -lm
OMP_NUM_THREADS=5 numactl -N 1 --preferred=17 ./independent-implicit-only
for round in 1 2 3 4 5 6 7 8; do
    order=(control off on)
    [[ $((round%2)) == 0 ]] && order=(on off control)
    for mode in "${order[@]}"; do
        export TRSM_IMPLICIT_OWNER=0
        binary=implicit-only
        [[ $mode == control ]] && binary=control
        [[ $mode == on ]] && export TRSM_IMPLICIT_OWNER=1
        echo "IMPLICIT_ONLY_FORMAL round=$round mode=$mode"
        numactl -N 1 --preferred=17 "./$binary" 512 19968 11
    done
done
for mode in 0 1; do
    echo "IMPLICIT_ONLY_PROFILE mode=$mode"
    TRSM_IMPLICIT_OWNER=$mode TRSM_PROFILE=3 numactl -N 1 --preferred=17 ./implicit-only 512 19968 5
done
