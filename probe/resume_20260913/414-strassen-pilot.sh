#!/usr/bin/env bash
set -eo pipefail
cd generated
flags=(-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp)
export GLIBC_TUNABLES=glibc.malloc.hugetlb=1
# Separate OpenMP materialization phases from KML's own execution without idle spinning.
export OMP_WAIT_POLICY=passive GOMP_SPINCOUNT=0
gcc "${flags[@]}" strassen_pilot.c -o strassen-pilot -lm -lkblas
for shape in '64 96 80' '65 97 81' '256 19968 256' '1216 17024 1216' '8512 512 8512'; do
    read -r m n k <<<"$shape"
    echo "STRASSEN_PILOT shape=$shape"
    numactl -N 1 --preferred=17 ./strassen-pilot "$m" "$n" "$k" 2
done
