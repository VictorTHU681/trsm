#!/usr/bin/env bash
set -eo pipefail
cd generated
flags=(-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp)
export GLIBC_TUNABLES=glibc.malloc.hugetlb=1
gcc "${flags[@]}" fold-tile-boundary.c -o fold-tile-boundary -lm
OMP_NUM_THREADS=1 numactl -N 1 --preferred=17 ./fold-tile-boundary
gcc "${flags[@]}" independent-fold.c trsm-fold-tile.c -o independent-fold-tile -lm
OMP_NUM_THREADS=5 TRSM_FOLD_TILE=1 TRSM_FOLD_PREF=0 numactl -N 1 --preferred=17 ./independent-fold-tile
gcc "${flags[@]}" bench_trsm.c trsm-fold-tile.c -o fold-tile -lm -lkblas
for round in 1 2 3; do
    order=(control off row tile nt tile_b tile_a)
    [[ $((round%2)) == 0 ]] && order=(tile_a tile_b nt tile row off control)
    for mode in "${order[@]}"; do
        binary=fold-tile
        export TRSM_FOLD_OWNER=1 TRSM_FOLD_KERNEL=3 TRSM_FOLD_PREF=0 TRSM_FOLD_TILE=1
        [[ $mode == control ]] && binary=control
        [[ $mode == off ]] && export TRSM_FOLD_OWNER=0
        [[ $mode == row ]] && export TRSM_FOLD_TILE=0
        [[ $mode == nt ]] && export TRSM_FOLD_TILE=2
        [[ $mode == tile_b ]] && export TRSM_FOLD_PREF=1
        [[ $mode == tile_a ]] && export TRSM_FOLD_PREF=2
        echo "FOLD54_TILE round=$round mode=$mode m=512 n=19968"
        numactl -N 1 --preferred=17 "./$binary" 512 19968 7
    done
done
for mode in row tile nt; do
    export TRSM_FOLD_TILE=1 TRSM_FOLD_PREF=0
    [[ $mode == row ]] && export TRSM_FOLD_TILE=0
    [[ $mode == nt ]] && export TRSM_FOLD_TILE=2
    echo "FOLD54_PROFILE mode=$mode m=512 n=19968"
    TRSM_FOLD_OWNER=1 TRSM_PROFILE=3 numactl -N 1 --preferred=17 ./fold-tile 512 19968 5
done
sha256sum trsm-fold-tile.c fold-tile-boundary.c
