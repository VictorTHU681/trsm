#!/usr/bin/env bash
set -eo pipefail
cd generated
flags=(-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp)
export GLIBC_TUNABLES=glibc.malloc.hugetlb=1
gcc "${flags[@]}" bench_trsm.c fullinv.c -o fullinv -lm -lkblas
for round in 1 2 3; do
    order=(0 32 64 128)
    [[ $round == 2 ]] && order=(128 64 32 0)
    for nc in "${order[@]}"; do
        echo "INVERSE_SCREEN round=$round nc=$nc m=512 n=19968"
        if (( nc == 0 )); then
            TRSM_FULLINV=0 numactl -N 1 --preferred=17 ./fullinv 512 19968 5 | tee "inv-screen-$round-$nc.log"
        else
            TRSM_FULLINV_NC=$nc TRSM_FULLINV_PROFILE=1 numactl -N 1 --preferred=17 ./fullinv 512 19968 5 | tee "inv-screen-$round-$nc.log"
        fi
        grep -q 'PASS' "inv-screen-$round-$nc.log"
        if grep -q 'FAIL' "inv-screen-$round-$nc.log"; then exit 3; fi
    done
done
