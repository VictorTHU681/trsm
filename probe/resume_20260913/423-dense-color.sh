#!/usr/bin/env bash
set -eo pipefail
cd generated
flags=(-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp)
export GLIBC_TUNABLES=glibc.malloc.hugetlb=1 STRASSEN_MAXDEPTH=0 DENSE_MB=192
unset DENSE_ORDER
gcc "${flags[@]}" strassen-check.c gemm-duplex-color.c -o dense-color -lm -lkblas
for mb in 64 128 192 256; do
    echo "DENSE_COLOR_CHECK mb=$mb"
    DENSE_MB=$mb DENSE_ASHIFT=4096 DENSE_BSHIFT=8192 DENSE_CSHIFT=16384 numactl -N 1 --preferred=17 ./dense-color 208 160 176 1
done
for bshift in 0 64 256 512 1024 4096 8192 16384 32768; do
    echo "DENSE_COLOR ashift=0 bshift=$bshift cshift=0"
    DENSE_BSHIFT=$bshift numactl -N 1 --preferred=17 ./dense-color 1216 17024 1216 2
done
for shifts in '64 128 0' '4096 8192 0' '16384 32768 0' '0 0 64' '0 0 4096' '0 0 16384' '0 0 32768'; do
    read -r ashift bshift cshift <<<"$shifts"
    echo "DENSE_COLOR ashift=$ashift bshift=$bshift cshift=$cshift"
    DENSE_ASHIFT=$ashift DENSE_BSHIFT=$bshift DENSE_CSHIFT=$cshift numactl -N 1 --preferred=17 ./dense-color 1216 17024 1216 2
done
