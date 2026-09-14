#!/usr/bin/env bash
set -eo pipefail
cd generated
export GLIBC_TUNABLES=glibc.malloc.hugetlb=1
flags=(-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp)
gcc "${flags[@]}" bench_trsm.c tiled_right.c -o tiled-right -lm -lkblas
gcc "${flags[@]}" independent-fullinv.c tiled_right.c -o tiled-known -lm
OMP_NUM_THREADS=5 numactl -N 1 --preferred=17 ./tiled-known
for shape in '512 19968' '2432 17024' '17024 512'; do
    read -r m n <<<"$shape"
    for nb in 128 256 512; do
        for nc in 32 128; do
            echo "TILED_RIGHT_FORMAL m=$m n=$n nb=$nb nc=$nc"
            TRSM_TILE_PROFILE=1 TRSM_TILE_NB=$nb TRSM_TILE_NC=$nc numactl -N 1 --preferred=17 ./tiled-right "$m" "$n" 3
        done
    done
done
sha256sum tiled_right.c right-update.h fullinv-compact.h trsm-v55.c bench_trsm.c
echo TILED_RIGHT_DONE
