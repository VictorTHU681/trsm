#!/usr/bin/env bash
set -eo pipefail
cd generated
export GLIBC_TUNABLES=glibc.malloc.hugetlb=1
gcc -O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp fp64_product.c -o fp64-product -lm
for threads in 1 38; do
    echo "FP64_THREADS=$threads"
    OMP_NUM_THREADS=$threads numactl -N 1 --preferred=17 ./fp64-product 32
    OMP_NUM_THREADS=$threads numactl -N 1 --preferred=17 ./fp64-product 512
done
for shape in '512 19968' '2432 17024' '17024 512'; do
    read -r m n <<<"$shape"
    echo "CONTROL_PROFILE m=$m n=$n"
    TRSM_PROFILE=2 numactl -N 1 --preferred=17 ./control "$m" "$n" 3
done
sha256sum fp64_product.c digits.c trsm-v55.c
echo PROFILE_CONTROL_DONE
