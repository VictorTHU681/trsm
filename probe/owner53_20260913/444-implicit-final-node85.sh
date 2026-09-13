#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname "$0")"
source ./env.sh
cd generated
flags=(-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp)
export GLIBC_TUNABLES=glibc.malloc.hugetlb=1
gcc "${flags[@]}" bench_trsm.c trsm-v53.c -o control -lm -lkblas
gcc "${flags[@]}" bench_trsm.c trsm-implicit-final.c -o implicit-final -lm -lkblas
for shape in '512 19968'; do
    read -r m n <<<"$shape"
    rounds=4;[[ $m == 512 ]] && rounds=12
    for ((round=1;round<=rounds;round++)); do
        order=(control final off)
        [[ $((round%2)) == 0 ]] && order=(off final control)
        for mode in "${order[@]}"; do
            unset TRSM_IMPLICIT_OWNER
            binary=implicit-final
            [[ $mode == control ]] && binary=control
            [[ $mode == off ]] && export TRSM_IMPLICIT_OWNER=0
            echo "IMPLICIT_FINAL_FORMAL round=$round mode=$mode m=$m n=$n"
            numactl -N 1 --preferred=17 "./$binary" "$m" "$n" 11
        done
    done
done
sha256sum bench_trsm.c trsm-v53.c trsm-implicit-final.c
echo IMPLICIT_FINAL_CROSSNODE_DONE
