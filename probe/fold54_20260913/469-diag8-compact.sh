#!/usr/bin/env bash
set -eo pipefail
cd generated
flags=(-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp)
export GLIBC_TUNABLES=glibc.malloc.hugetlb=1
gcc "${flags[@]}" -DTRSM_DIAG8_CPF=0 -c trsm-diag8-compact.c -o diag8-compact.o
gcc "${flags[@]}" independent-diag8.c diag8-compact.o -o independent-diag8-compact -lm
OMP_NUM_THREADS=5 numactl -N 1 --preferred=17 ./independent-diag8-compact
gcc "${flags[@]}" bench_trsm.c diag8-compact.o -o diag8-compact -lm -lkblas
gcc "${flags[@]}" -DTRSM_DIAG8_CPF=1 bench_trsm.c trsm-diag8-compact.c -o diag8-cpf -lm -lkblas
for round in 1 2 3 4 5 6; do
    order=(control off old compact cpf)
    [[ $((round%2)) == 0 ]] && order=(cpf compact old off control)
    for mode in "${order[@]}"; do
        binary=diag8-compact
        export TRSM_DIAG8=1
        case $mode in
            control) binary=control;export TRSM_DIAG8=0;;
            off) export TRSM_DIAG8=0;;
            old) binary=diag8;;
            cpf) binary=diag8-cpf;;
        esac
        echo "DIAG8_COMPACT_FORMAL round=$round mode=$mode m=512 n=19968"
        numactl -N 1 --preferred=17 "./$binary" 512 19968 13
    done
done
for mode in off compact cpf; do
    binary=diag8-compact
    export TRSM_DIAG8=1
    [[ $mode == off ]] && export TRSM_DIAG8=0
    [[ $mode == cpf ]] && binary=diag8-cpf
    echo "DIAG8_COMPACT_PROFILE mode=$mode m=512 n=19968"
    TRSM_PROFILE=3 numactl -N 1 --preferred=17 "./$binary" 512 19968 5
done
sha256sum trsm-diag8-compact.c
