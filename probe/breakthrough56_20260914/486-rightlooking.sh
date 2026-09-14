#!/usr/bin/env bash
set -eo pipefail
cd generated
export GLIBC_TUNABLES=glibc.malloc.hugetlb=1
flags=(-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp)
gcc "${flags[@]}" bench_trsm.c rightlooking.c -o rightlooking -lm -lkblas
gcc "${flags[@]}" independent-fullinv.c rightlooking.c -o right-known -lm
OMP_NUM_THREADS=5 numactl -N 1 --preferred=17 ./right-known
for shape in '512 19968' '2432 17024' '17024 512'; do
    read -r m n <<<"$shape"
    echo "RIGHT_CONTROL m=$m n=$n"
    numactl -N 1 --preferred=17 ./control "$m" "$n" 3
    for nb in 128 256 512; do
        for nc in 32 128; do
            echo "RIGHT_FORMAL m=$m n=$n nb=$nb nc=$nc"
            TRSM_RIGHT_PROFILE=1 TRSM_RIGHT_NB=$nb TRSM_RIGHT_NC=$nc numactl -N 1 --preferred=17 ./rightlooking "$m" "$n" 3
        done
    done
done
sha256sum rightlooking.c right-kernel.h trsm-v55.c bench_trsm.c
echo RIGHTLOOKING_DONE
