#!/usr/bin/env bash
set -eo pipefail
cd generated
flags=(-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp)
export GLIBC_TUNABLES=glibc.malloc.hugetlb=1
sha256sum trsm-ownerdiag58.c independent-owner58.c
for mode in 0 1 2; do
    gcc "${flags[@]}" -DTRSM_OWNER_DIAG58=$mode bench_trsm.c trsm-ownerdiag58.c -o owner58-$mode -lm -lkblas
    nm -S "owner58-$mode" | grep -E 'run_task_owner_cursor|ownerdiag58_plain'
done
gcc "${flags[@]}" independent-owner58.c -o independent-owner58 -lm
OMP_NUM_THREADS=5 numactl -N 1 --preferred=17 ./independent-owner58
for round in 1 2 3 4 5 6; do
    order=(control off outline nocpf)
    [[ $((round%2)) == 0 ]] && order=(nocpf outline off control)
    for mode in "${order[@]}"; do
        case "$mode" in control) binary=control;; off) binary=owner58-0;; outline) binary=owner58-1;; nocpf) binary=owner58-2;; esac
        echo "OWNER58_FORMAL round=$round mode=$mode m=512 n=19968"
        numactl -N 1 --preferred=17 "./$binary" 512 19968 11
    done
done
for mode in 0 1 2; do
    echo "OWNER58_PROFILE mode=$mode m=512 n=19968"
    TRSM_PROFILE=2 numactl -N 1 --preferred=17 "./owner58-$mode" 512 19968 3
done
echo OWNER58_DONE
