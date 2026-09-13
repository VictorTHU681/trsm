#!/usr/bin/env bash
set -eo pipefail
cd generated
flags=(-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp)
export GLIBC_TUNABLES=glibc.malloc.hugetlb=1
gcc "${flags[@]}" bench_trsm.c trsm-control.c -o front-control -lm -lkblas
for window in 16 32 48 64; do
    gcc "${flags[@]}" "-DPROBE_SOURCE=\"trsm-front$window.c\"" front-boundary.c -o "front-boundary$window" -lm
    echo "FRONT_BOUNDARY window=$window"
    OMP_NUM_THREADS=1 numactl -N 1 --preferred=17 taskset -c 40 "./front-boundary$window"
    gcc "${flags[@]}" bench_trsm.c "trsm-front$window.c" -o "front$window" -lm -lkblas
done
for round in 1 2 3 4; do
    order=(control 16 32 48 64)
    [[ $((round%2)) == 0 ]] && order=(64 48 32 16 control)
    for window in "${order[@]}"; do
        echo "FRONT_FORMAL round=$round window=$window"
        binary="front$window";[[ $window == control ]] && binary=front-control
        numactl -N 1 --preferred=17 "./$binary" 512 19968 11
    done
done
for window in control 16 32 48; do
    echo "FRONT_PROFILE window=$window"
    binary="front$window";[[ $window == control ]] && binary=front-control
    TRSM_PROFILE=2 numactl -N 1 --preferred=17 "./$binary" 512 19968 5
done
