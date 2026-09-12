#!/usr/bin/env bash
set -eo pipefail
cd generated
flags=(-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp)
gcc "${flags[@]}" micro-data.c -o micro-data -lm
for round in 1 2 3; do
    for mode in 0 1 2; do
        echo "DATA_MICRO round=$round data=$mode"
        MICRO_DENSE_DATA=$mode numactl -N 1 --preferred=17 taskset -c 40 ./micro-data
    done
done
