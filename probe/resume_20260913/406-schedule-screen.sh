#!/usr/bin/env bash
set -eo pipefail
cd generated
flags=(-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp)
for mode in control pfspread spread front32 late32; do
    echo "COMPILE $mode"
    gcc "${flags[@]}" "boundary-$mode.c" -o "boundary-$mode" -lm
    gcc "${flags[@]}" "micro-$mode.c" -o "micro-$mode" -lm
    numactl -N 1 --preferred=17 taskset -c 40 "./boundary-$mode"
done
for round in 1 2 3; do
    order=(control pfspread spread front32 late32)
    [[ $round == 2 ]] && order=(late32 front32 spread pfspread control)
    for mode in "${order[@]}"; do
        echo "MICRO round=$round mode=$mode"
        numactl -N 1 --preferred=17 taskset -c 40 "./micro-$mode"
    done
done
