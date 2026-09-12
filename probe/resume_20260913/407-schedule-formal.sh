#!/usr/bin/env bash
set -eo pipefail
cd generated
flags=(-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp)
export GLIBC_TUNABLES=glibc.malloc.hugetlb=1
for mode in control pfspread spread front32 late32; do
    echo "COMPILE_FORMAL $mode"
    gcc "${flags[@]}" bench_trsm.c "trsm-$mode.c" -o "bench-$mode" -lm -lkblas
done
for round in 1 2 3 4; do
    order=(control pfspread spread front32 late32)
    (( round % 2 == 0 )) && order=(late32 front32 spread pfspread control)
    for mode in "${order[@]}"; do
        echo "FORMAL round=$round mode=$mode m=512 n=19968 repeats=7"
        numactl -N 1 --preferred=17 "./bench-$mode" 512 19968 7 | tee "formal-$mode-$round.log"
        grep -q 'PASS' "formal-$mode-$round.log"
        if grep -q 'FAIL' "formal-$mode-$round.log"; then exit 3; fi
    done
done
