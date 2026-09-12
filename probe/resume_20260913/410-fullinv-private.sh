#!/usr/bin/env bash
set -eo pipefail
cd generated
flags=(-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp)
export GLIBC_TUNABLES=glibc.malloc.hugetlb=1
gcc "${flags[@]}" bench_trsm.c fullinv-v2.c -o fullinv-v2 -lm -lkblas
for round in 1 2 3; do
    order=(0 32 64 128)
    [[ $round == 2 ]] && order=(128 64 32 0)
    for nc in "${order[@]}"; do
        echo "INVERSE_PRIVATE round=$round nc=$nc m=512 n=19968"
        if (( nc == 0 )); then
            TRSM_FULLINV=0 numactl -N 1 --preferred=17 ./fullinv-v2 512 19968 5 | tee "inv-private-$round-$nc.log"
        else
            TRSM_FULLINV_NC=$nc TRSM_FULLINV_PROFILE=1 numactl -N 1 --preferred=17 ./fullinv-v2 512 19968 5 | tee "inv-private-$round-$nc.log"
        fi
        grep -q 'PASS' "inv-private-$round-$nc.log"
        if grep -q 'FAIL' "inv-private-$round-$nc.log"; then exit 3; fi
    done
done
for private in 0 1; do
    echo "PRIVATE_SW_ONLY private=$private"
    TRSM_FULLINV_PRIVATE=$private TRSM_FULLINV_NC=128 TRSM_FULLINV_PROFILE=1 numactl -N 1 --preferred=17 ./fullinv-v2 512 19968 5
done
