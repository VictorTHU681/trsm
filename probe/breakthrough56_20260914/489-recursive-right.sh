#!/usr/bin/env bash
set -eo pipefail
cd generated
export GLIBC_TUNABLES=glibc.malloc.hugetlb=1
flags=(-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp)
gcc "${flags[@]}" bench_trsm.c tiled-recursive.c -o tiled-recursive -lm -lkblas
gcc "${flags[@]}" independent-fullinv.c tiled-recursive.c -o recursive-known -lm
OMP_NUM_THREADS=5 numactl -N 1 --preferred=17 ./recursive-known
for shape in '512 19968' '2432 17024' '17024 512'; do
    read -r m n <<<"$shape"
    for settings in '128 128 0' '512 128 0' '256 128 1' '512 128 1' '512 64 1' '1024 64 1'; do
        read -r nb mb jo <<<"$settings"
        echo "RECURSIVE_RIGHT_FORMAL m=$m n=$n nb=$nb mb=$mb jouter=$jo"
        TRSM_TILE_PROFILE=1 TRSM_TILE_NB=$nb TRSM_TILE_MB=$mb TRSM_TILE_JOUTER=$jo numactl -N 1 --preferred=17 ./tiled-recursive "$m" "$n" 3
    done
done
sha256sum tiled-recursive.c block_inverse.h right-update.h fullinv-compact.h trsm-v55.c bench_trsm.c
echo RECURSIVE_RIGHT_DONE
