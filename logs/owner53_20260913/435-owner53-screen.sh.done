#!/usr/bin/env bash
set -eo pipefail
cd generated
flags=(-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp)
export GLIBC_TUNABLES=glibc.malloc.hugetlb=1
gcc "${flags[@]}" bench_trsm.c trsm-v53.c -o control -lm -lkblas
gcc "${flags[@]}" bench_trsm.c trsm-owner53.c -o owner53 -lm -lkblas
gcc "${flags[@]}" bounds_check.c -o bounds-check -lm
OMP_NUM_THREADS=1 numactl -N 1 --preferred=17 ./bounds-check
gcc "${flags[@]}" independent-owner53.c trsm-owner53.c -o independent-owner53 -lm
OMP_NUM_THREADS=5 numactl -N 1 --preferred=17 ./independent-owner53
for round in 1 2 3 4 5 6; do
    order=(control off implicit stack both)
    [[ $((round%2)) == 0 ]] && order=(both stack implicit off control)
    for mode in "${order[@]}"; do
        export TRSM_IMPLICIT_OWNER=0 TRSM_STACK_SCHED=0
        binary=owner53
        case "$mode" in
            control) binary=control;;
            implicit) export TRSM_IMPLICIT_OWNER=1;;
            stack) export TRSM_STACK_SCHED=1;;
            both) export TRSM_IMPLICIT_OWNER=1 TRSM_STACK_SCHED=1;;
        esac
        echo "OWNER53_FORMAL round=$round mode=$mode"
        numactl -N 1 --preferred=17 "./$binary" 512 19968 11
    done
done
for mode in off implicit stack both; do
    export TRSM_IMPLICIT_OWNER=0 TRSM_STACK_SCHED=0
    case "$mode" in
        implicit) export TRSM_IMPLICIT_OWNER=1;;
        stack) export TRSM_STACK_SCHED=1;;
        both) export TRSM_IMPLICIT_OWNER=1 TRSM_STACK_SCHED=1;;
    esac
    echo "OWNER53_PROFILE mode=$mode"
    TRSM_PROFILE=3 numactl -N 1 --preferred=17 ./owner53 512 19968 5
done
