#!/usr/bin/env bash
set -eo pipefail
cd generated
flags=(-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp)
export GLIBC_TUNABLES=glibc.malloc.hugetlb=1
objcopy -O binary --only-section=.text front-control front-control.text
objcopy -O binary --only-section=.text front64 front64.text
cmp front-control.text front64.text
echo FRONT64_TEXT_IDENTICAL_PASS
sha256sum front-control.text front64.text
gcc "${flags[@]}" independent-front48.c trsm-front48-audit.c -o independent-front48 -lm
OMP_NUM_THREADS=5 numactl -N 1 --preferred=17 ./independent-front48
for round in 1 2 3 4 5 6 7 8; do
    order=(control 48 64)
    [[ $((round%2)) == 0 ]] && order=(64 48 control)
    for window in "${order[@]}"; do
        echo "FRONT_RECHECK round=$round window=$window"
        binary="front$window";[[ $window == control ]] && binary=front-control
        numactl -N 1 --preferred=17 "./$binary" 512 19968 11
    done
done
