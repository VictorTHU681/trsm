#!/usr/bin/env bash
set -eo pipefail
cd generated
export GLIBC_TUNABLES=glibc.malloc.hugetlb=1
gcc -O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp digits.c -o digits -lm
for mode in 0 1 2; do
    OMP_NUM_THREADS=1 numactl -N 1 --preferred=17 ./digits 32 512 "$mode"
done
for threads in 1 38; do
    echo "DIGIT_THREADS=$threads"
    OMP_NUM_THREADS=$threads numactl -N 1 --preferred=17 ./digits 512 2048 0
done
sha256sum digits.c digit-kernels.h trsm-v55.c
echo DIGIT_SCREEN_DONE
