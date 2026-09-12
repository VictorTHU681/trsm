#!/usr/bin/env bash
set -eo pipefail
cd generated
flags=(-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp)
gcc "${flags[@]}" fullinv-boundary.c -o fullinv-boundary -lm
numactl -N 1 --preferred=17 taskset -c 40 ./fullinv-boundary
gcc "${flags[@]}" independent-fullinv.c fullinv-v3-audit.c -o independent-fullinv -lm
numactl -N 1 --preferred=17 ./independent-fullinv
