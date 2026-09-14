#!/usr/bin/env bash
set -eo pipefail
cd generated
export GLIBC_TUNABLES=glibc.malloc.hugetlb=1
gcc -O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp residue_fused.c -o residue-fused -lm
for mode in 0 1 2; do
    OMP_NUM_THREADS=1 numactl -N 1 --preferred=17 ./residue-fused 32 512 "$mode"
done
for m in 512 1024; do
    echo "RNS_THREADS=38 M=$m"
    numactl -N 1 --preferred=17 ./residue-fused "$m" 2048 0
done
sha256sum residue_fused.c rns13-fixed-constants.h digit-kernels.h trsm-v55.c
echo RESIDUE_FUSED_DONE
