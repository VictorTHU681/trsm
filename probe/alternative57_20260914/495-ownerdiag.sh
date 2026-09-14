#!/usr/bin/env bash
set -eo pipefail
cd generated
flags=(-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp)
export GLIBC_TUNABLES=glibc.malloc.hugetlb=1
sha256sum trsm-ownerdiag.c independent-ownerdiag.c
gcc "${flags[@]}" independent-ownerdiag.c -o independent-ownerdiag -lm
OMP_NUM_THREADS=5 numactl -N 1 --preferred=17 ./independent-ownerdiag
gcc "${flags[@]}" bench_trsm.c trsm-ownerdiag.c -o ownerdiag -lm -lkblas
for round in 1 2 3 4 5 6; do
    case "$round" in
        1) order=(control off on);; 2) order=(on off control);;
        3) order=(off on control);; 4) order=(control on off);;
        5) order=(on control off);; 6) order=(off control on);;
    esac
    for mode in "${order[@]}"; do
        binary=ownerdiag
        export TRSM_OWNER_DIAG_SIMPLE=0
        case "$mode" in control) binary=control;; on) export TRSM_OWNER_DIAG_SIMPLE=1;; esac
        echo "OWNERDIAG_FORMAL round=$round mode=$mode m=512 n=19968"
        numactl -N 1 --preferred=17 "./$binary" 512 19968 11
    done
done
for version in 0 1; do
    echo "OWNERDIAG_PROFILE version=$version m=512 n=19968"
    TRSM_OWNER_DIAG_SIMPLE=$version TRSM_PROFILE=2 numactl -N 1 --preferred=17 ./ownerdiag 512 19968 3
done
echo OWNERDIAG_DONE
