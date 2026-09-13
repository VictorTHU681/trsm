#!/usr/bin/env bash
set -eo pipefail
cd generated
flags=(-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp)
export GLIBC_TUNABLES=glibc.malloc.hugetlb=1
gcc "${flags[@]}" independent-fold.c trsm-noinstr.c -o independent-noinstr-owner -lm
OMP_NUM_THREADS=5 TRSM_NOINSTR=1 numactl -N 1 --preferred=17 ./independent-noinstr-owner
gcc "${flags[@]}" independent-duplex.c trsm-noinstr.c -o independent-noinstr-duplex -lm
OMP_NUM_THREADS=5 TRSM_NOINSTR=1 TRSM_OWNER=0 TRSM_ANT=1 TRSM_DIRECT=0 TRSM_MB=256 TRSM_SWHOST=0 numactl -N 1 --preferred=17 ./independent-noinstr-duplex
OMP_NUM_THREADS=5 TRSM_NOINSTR=1 TRSM_OWNER=0 TRSM_ANT=1 TRSM_DIRECT=0 TRSM_MB=192 TRSM_JIT_MB=0 TRSM_PPMODE=1 numactl -N 1 --preferred=17 ./independent-noinstr-duplex
gcc "${flags[@]}" bench_trsm.c trsm-noinstr.c -o noinstr -lm -lkblas
for shape in '512 19968' '2432 17024' '17024 512'; do
    read -r m n <<<"$shape"
    for round in 1 2 3 4 5 6; do
        order=(control off on)
        [[ $((round%2)) == 0 ]] && order=(on off control)
        for mode in "${order[@]}"; do
            binary=noinstr
            export TRSM_NOINSTR=0
            [[ $mode == control ]] && binary=control
            [[ $mode == on ]] && export TRSM_NOINSTR=1
            echo "NOINSTR_FORMAL round=$round mode=$mode m=$m n=$n"
            numactl -N 1 --preferred=17 "./$binary" "$m" "$n" 9
        done
    done
done
sha256sum trsm-noinstr.c
