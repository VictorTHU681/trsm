#!/usr/bin/env bash
set -eo pipefail
cd generated
flags=(-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp)
export GLIBC_TUNABLES=glibc.malloc.hugetlb=1
gcc "${flags[@]}" diag_check.c -o diag-check -lm
OMP_NUM_THREADS=1 numactl -N 1 --preferred=17 taskset -c 40 ./diag-check
gcc "${flags[@]}" bench_trsm.c trsm-diag-order.c -o diag-order -lm -lkblas
for round in 1 2 3 4; do
    order=(control 0 1 2 3)
    [[ $((round%2)) == 0 ]] && order=(3 2 1 0 control)
    for shape in '512 19968' '2432 17024' '17024 512'; do
        read -r m n <<<"$shape"
        for mode in "${order[@]}"; do
            export TRSM_DIAG_ORDER=0
            binary=diag-order
            if [[ $mode == control ]]; then binary=control;else export TRSM_DIAG_ORDER=$mode;fi
            echo "DIAG_FORMAL round=$round mode=$mode m=$m n=$n"
            numactl -N 1 --preferred=17 "./$binary" "$m" "$n" 7
        done
    done
done
for mode in 0 1 2 3; do
    for shape in '512 19968' '2432 17024' '17024 512'; do
        read -r m n <<<"$shape"
        echo "DIAG_PROFILE mode=$mode m=$m n=$n"
        TRSM_DIAG_ORDER=$mode TRSM_PROFILE=3 numactl -N 1 --preferred=17 ./diag-order "$m" "$n" 3
    done
done
