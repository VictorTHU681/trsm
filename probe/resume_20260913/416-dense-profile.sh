#!/usr/bin/env bash
set -eo pipefail
cd generated
flags=(-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp)
export GLIBC_TUNABLES=glibc.malloc.hugetlb=1 STRASSEN_MAXDEPTH=0
gcc "${flags[@]}" strassen-profile.c gemm-duplex-profile.c -o dense-profile -lm -lkblas
for shape in '1216 17024 1216' '8512 512 8512'; do
    read -r m n k <<<"$shape"
    numactl -N 1 --preferred=17 ./dense-profile "$m" "$n" "$k" 3
done
