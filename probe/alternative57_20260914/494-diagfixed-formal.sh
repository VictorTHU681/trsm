#!/usr/bin/env bash
set -eo pipefail
cd generated
flags=(-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp)
export GLIBC_TUNABLES=glibc.malloc.hugetlb=1
gcc "${flags[@]}" bench_trsm.c trsm-v55.c -o control -lm -lkblas
gcc "${flags[@]}" bench_trsm.c trsm-diagfixed.c -o diagfixed -lm -lkblas
for round in 1 2 3 4; do
    order=(control off fixed nosw)
    [[ $((round%2)) == 0 ]] && order=(nosw fixed off control)
    for mode in "${order[@]}"; do
        binary=diagfixed
        export TRSM_DIAG_FIXED=0
        case "$mode" in control) binary=control;; fixed) export TRSM_DIAG_FIXED=1;; nosw) export TRSM_DIAG_FIXED=2;; esac
        echo "DIAGFIXED_FORMAL round=$round mode=$mode m=2432 n=17024"
        numactl -N 1 --preferred=17 "./$binary" 2432 17024 7
    done
done
for shape in '512 19968' '17024 512'; do
    read -r m n <<<"$shape"
    for round in 1 2 3; do
        order=(control off nosw)
        [[ $((round%2)) == 0 ]] && order=(nosw off control)
        for mode in "${order[@]}"; do
            binary=diagfixed
            export TRSM_DIAG_FIXED=0
            case "$mode" in control) binary=control;; nosw) export TRSM_DIAG_FIXED=2;; esac
            echo "DIAGFIXED_FORMAL round=$round mode=$mode m=$m n=$n"
            numactl -N 1 --preferred=17 "./$binary" "$m" "$n" 7
        done
    done
done
for version in 0 1 2; do
    echo "DIAGFIXED_PROFILE version=$version m=2432 n=17024"
    TRSM_DIAG_FIXED=$version TRSM_PROFILE=2 numactl -N 1 --preferred=17 ./diagfixed 2432 17024 3
done
sha256sum trsm-diagfixed.c independent-diagfixed.c trsm-v55.c bench_trsm.c
echo DIAGFIXED_FORMAL_DONE
