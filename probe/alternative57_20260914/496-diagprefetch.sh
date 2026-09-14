#!/usr/bin/env bash
set -eo pipefail
cd generated
flags=(-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp)
export GLIBC_TUNABLES=glibc.malloc.hugetlb=1
sha256sum trsm-diagfixed-prefetch.c trsm-ownerdiag-prefetch.c
gcc "${flags[@]}" bench_trsm.c trsm-diagfixed-prefetch.c -o diagfixed-prefetch -lm -lkblas
gcc "${flags[@]}" bench_trsm.c trsm-ownerdiag-prefetch.c -o ownerdiag-prefetch -lm -lkblas
for shape in '512 19968' '2432 17024'; do
    read -r m n <<<"$shape"
    for round in 1 2 3 4 5 6; do
        order=(control off fixed bounded)
        [[ $((round%2)) == 0 ]] && order=(bounded fixed off control)
        for mode in "${order[@]}"; do
            export TRSM_OWNER_DIAG_SIMPLE=0 TRSM_DIAG_FIXED=0
            if [[ $m == 512 ]]; then
                binary=ownerdiag-prefetch
                case "$mode" in control) binary=control;; fixed) export TRSM_OWNER_DIAG_SIMPLE=1;; bounded) export TRSM_OWNER_DIAG_SIMPLE=2;; esac
            else
                binary=diagfixed-prefetch
                case "$mode" in control) binary=control;; fixed) export TRSM_DIAG_FIXED=1;; bounded) export TRSM_DIAG_FIXED=2;; esac
            fi
            echo "DIAGPREFETCH_FORMAL round=$round mode=$mode m=$m n=$n"
            numactl -N 1 --preferred=17 "./$binary" "$m" "$n" 11
        done
    done
done
for mode in 0 1 2; do
    echo "DIAGPREFETCH_PROFILE version=$mode m=512 n=19968"
    TRSM_OWNER_DIAG_SIMPLE=$mode TRSM_PROFILE=2 numactl -N 1 --preferred=17 ./ownerdiag-prefetch 512 19968 3
    echo "DIAGPREFETCH_PROFILE version=$mode m=2432 n=17024"
    TRSM_DIAG_FIXED=$mode TRSM_PROFILE=2 numactl -N 1 --preferred=17 ./diagfixed-prefetch 2432 17024 3
done
echo DIAGPREFETCH_DONE
