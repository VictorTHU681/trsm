#!/usr/bin/env bash
set -eo pipefail
cd generated
# Built by 423 from strassen-check.c and gemm-duplex-color.c.
export GLIBC_TUNABLES=glibc.malloc.hugetlb=1 STRASSEN_MAXDEPTH=2 DENSE_MB=128 DENSE_ORDER=1
unset DENSE_ASHIFT DENSE_BSHIFT DENSE_CSHIFT
for shape in '208 160 176' '1216 17024 1216' '8512 512 8512'; do
    read -r m n k <<<"$shape"
    echo "STRASSEN_REBLOCKED m=$m n=$n k=$k mb=128 order=1"
    numactl -N 1 --preferred=17 ./dense-color "$m" "$n" "$k" 2
done
