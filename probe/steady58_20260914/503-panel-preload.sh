#!/usr/bin/env bash
set -eo pipefail
cd generated
flags=(-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp)
export GLIBC_TUNABLES=glibc.malloc.hugetlb=1
sha256sum trsm-preload58.c independent-preload58.c
gcc "${flags[@]}" independent-preload58.c -o independent-preload58 -lm
for mode in 0 1; do
    OMP_NUM_THREADS=5 TRSM_PRELOAD58=$mode numactl -N 1 --preferred=17 ./independent-preload58
done
gcc "${flags[@]}" bench_trsm.c trsm-preload58.c -o preload58 -lm -lkblas
for round in 1 2 3 4 5 6; do
    order=(control off preload prefetch)
    [[ $((round%2)) == 0 ]] && order=(prefetch preload off control)
    for mode in "${order[@]}"; do
        binary=preload58
        export TRSM_PRELOAD58=0 TRSM_LOADPF=0
        case "$mode" in control) binary=control;; preload) export TRSM_PRELOAD58=1;; prefetch) export TRSM_PRELOAD58=1 TRSM_LOADPF=16;; esac
        echo "PRELOAD58_FORMAL round=$round mode=$mode m=512 n=19968"
        numactl -N 1 --preferred=17 "./$binary" 512 19968 11
    done
done
for mode in 0 1; do
    echo "PRELOAD58_PROFILE mode=$mode m=512 n=19968"
    TRSM_PRELOAD58=$mode TRSM_LOADPF=0 TRSM_PROFILE=2 numactl -N 1 --preferred=17 ./preload58 512 19968 3
done
echo PRELOAD58_DONE
