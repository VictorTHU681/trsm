#!/usr/bin/env bash
set -eo pipefail
cd generated
flags=(-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp)
export GLIBC_TUNABLES=glibc.malloc.hugetlb=1
gcc "${flags[@]}" bench_trsm.c recursive_trsm.c -o recursive-trsm -lm -lkblas
for round in 1 2; do
    for shape in '512 19968' '2432 17024' '17024 512'; do
        read -r m n <<<"$shape"
        order=(-1 0 1)
        [[ $round == 2 ]] && order=(1 0 -1)
        for level in "${order[@]}"; do
            echo "RECURSIVE_FORMAL round=$round m=$m n=$n level=$level"
            if [[ $level == -1 ]]; then
                TRSM_FUSED=0 numactl -N 1 --preferred=17 ./recursive-trsm "$m" "$n" 5
            else
                TRSM_FUSED=1 TRSM_FUSED_LEVEL=$level numactl -N 1 --preferred=17 ./recursive-trsm "$m" "$n" 5
            fi
        done
    done
done
for shape in '512 19968' '2432 17024' '17024 512'; do
    read -r m n <<<"$shape"
    for level in 0 1; do
        echo "RECURSIVE_PROFILE m=$m n=$n level=$level"
        TRSM_FUSED=1 TRSM_FUSED_LEVEL=$level TRSM_FUSED_PROFILE=1 numactl -N 1 --preferred=17 ./recursive-trsm "$m" "$n" 3
    done
done
