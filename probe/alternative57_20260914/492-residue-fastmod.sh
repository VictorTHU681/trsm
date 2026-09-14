#!/usr/bin/env bash
set -eo pipefail
cd generated
export GLIBC_TUNABLES=glibc.malloc.hugetlb=1
gcc -O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp residue_fastmod.c -o residue-fastmod -lm
for k in 512 2048; do
    for mode in 0 1 2; do
        OMP_NUM_THREADS=1 numactl -N 1 --preferred=17 ./residue-fastmod 32 "$k" "$mode"
    done
done
for m in 512 1024; do
    echo "RNS_THREADS=38 M=$m"
    numactl -N 1 --preferred=17 ./residue-fastmod "$m" 2048 0
done
sha256sum residue_fastmod.c rns13-fixed-constants.h digit-kernels.h trsm-v55.c
echo RESIDUE_FASTMOD_DONE
