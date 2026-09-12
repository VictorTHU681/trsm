#!/usr/bin/env bash
set -eo pipefail
cd generated
flags=(-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp)
export GLIBC_TUNABLES=glibc.malloc.hugetlb=1
for mode in control pfspread spread front32 late32; do
    gcc "${flags[@]}" bench_trsm.c "trsm-$mode-pmu.c" -o "profile-$mode" -lm -lkblas
    echo "PROFILE mode=$mode m=512 n=19968 repeats=5"
    TRSM_PROFILE=2 numactl -N 1 --preferred=17 "./profile-$mode" 512 19968 5
done
