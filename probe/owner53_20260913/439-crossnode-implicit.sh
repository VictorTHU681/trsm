#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname "$0")"
source ./env.sh
cd generated
flags=(-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp)
export GLIBC_TUNABLES=glibc.malloc.hugetlb=1
gcc "${flags[@]}" bench_trsm.c trsm-v53.c -o control -lm -lkblas
gcc "${flags[@]}" bench_trsm.c trsm-implicit-only.c -o implicit-only -lm -lkblas
for round in 1 2 3 4 5 6 7 8 9 10 11 12; do
    order=(control off on)
    [[ $((round%2)) == 0 ]] && order=(on off control)
    for mode in "${order[@]}"; do
        export TRSM_IMPLICIT_OWNER=0
        binary=implicit-only
        [[ $mode == control ]] && binary=control
        [[ $mode == on ]] && export TRSM_IMPLICIT_OWNER=1
        echo "IMPLICIT_CROSSNODE round=$round mode=$mode m=512 n=19968"
        numactl -N 1 --preferred=17 "./$binary" 512 19968 11
    done
done
for round in 1 2 3; do
    order=(control on)
    [[ $((round%2)) == 0 ]] && order=(on control)
    for shape in '2432 17024' '17024 512'; do
        read -r m n <<<"$shape"
        for mode in "${order[@]}"; do
            binary=control;[[ $mode == on ]] && binary=implicit-only
            echo "IMPLICIT_CROSSNODE round=$round mode=$mode m=$m n=$n"
            TRSM_IMPLICIT_OWNER=1 TRSM_EXPECT_IMPLICIT=0 numactl -N 1 --preferred=17 "./$binary" "$m" "$n" 7
        done
    done
done
sha256sum bench_trsm.c trsm-v53.c trsm-implicit-only.c
echo IMPLICIT_CROSSNODE_DONE
