#!/usr/bin/env bash
set -eo pipefail
cd generated
export GLIBC_TUNABLES=glibc.malloc.hugetlb=1
gcc -O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp bench_trsm.c fullinv-blocked.c -o inverse-blocked -lm -lkblas
for shape in '512 19968' '2432 17024'; do
    read -r m n <<<"$shape"
    for nc in 32 128; do
        for kc in 64 128 256; do
            echo "BLOCKED_INV m=$m n=$n nc=$nc kc=$kc"
            TRSM_FULLINV_PROFILE=1 TRSM_FULLINV_NC=$nc TRSM_INV_KC=$kc numactl -N 1 --preferred=17 ./inverse-blocked "$m" "$n" 3
        done
    done
done
sha256sum fullinv-blocked.c fullinv-compact.h bench_trsm.c
echo BLOCKED_INVERSE_DONE
