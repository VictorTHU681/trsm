#!/usr/bin/env bash
set -eo pipefail
cd generated
flags=(-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp)
export GLIBC_TUNABLES=glibc.malloc.hugetlb=1
gcc "${flags[@]}" driver-direct.c direct.c -o direct-driver -lm -lkblas
for shape in '64 192 176' '224 320 192' '129 131 81'; do
    read -r m n k <<<"$shape"
    for alpha in -1 1; do
        for beta in 0 1; do
            numactl -N 1 --preferred=17 ./direct-driver "$m" "$n" "$k" 1 2 "$alpha" "$beta" 7
        done
    done
done
for exp in -400 400; do
    numactl -N 1 --preferred=17 ./direct-driver 224 320 192 1 2 -1 1 1 "$exp"
done
for shape in '256 19968 256' '1216 17024 1216' '8512 512 8512'; do
    read -r m n k <<<"$shape"
    numactl -N 1 --preferred=17 ./direct-driver "$m" "$n" "$k" 3 2 -1 1 64
done
echo DIRECT_BALANCED_MB112
FUSED_MB=112 numactl -N 1 --preferred=17 ./direct-driver 8512 512 8512 3 2 -1 1 64
