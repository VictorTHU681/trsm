#!/usr/bin/env bash
set -eo pipefail
cd generated
flags=(-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp)
export GLIBC_TUNABLES=glibc.malloc.hugetlb=1
for mr in 8 16 32 64; do
    gcc "${flags[@]}" bench_trsm.c "fullinv-mr$mr.c" -o "fullinv-mr$mr" -lm -lkblas
done
for round in 1 2; do
    order=(8 16 32 64)
    [[ $round == 2 ]] && order=(64 32 16 8)
    for mr in "${order[@]}"; do
        echo "INVERSE_SHAPE round=$round mr=$mr nc=128"
        TRSM_FULLINV_NC=128 TRSM_FULLINV_PROFILE=1 numactl -N 1 --preferred=17 "./fullinv-mr$mr" 512 19968 5 | tee "inv-shape-$round-$mr.log"
        grep -q 'PASS' "inv-shape-$round-$mr.log"
        if grep -q 'FAIL' "inv-shape-$round-$mr.log"; then exit 3; fi
    done
done
