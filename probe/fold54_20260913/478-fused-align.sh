#!/usr/bin/env bash
set -eo pipefail
cd generated
flags=(-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp)
export GLIBC_TUNABLES=glibc.malloc.hugetlb=1
for align in 64 1024; do
    gcc "${flags[@]}" -DTRSM_OWNER_ALIGN=$align bench_trsm.c trsm-fused-align.c -o fused-align-$align -lm -lkblas
    echo "FUSED_ALIGN_SYMBOL $align"
    nm -S fused-align-$align | rg 'run_task_owner_cursor' || nm -S fused-align-$align
done
orders=('control final a64 a1024' 'a1024 a64 final control' 'final control a1024 a64' 'a64 a1024 control final')
for round in 1 2 3 4 5 6 7 8; do
    read -ra order <<<"${orders[$(((round-1)%4))]}"
    for mode in "${order[@]}"; do
        binary=fused-final
        unset TRSM_FUSED_DIAG
        [[ $mode == control ]] && binary=control
        [[ $mode == a64 ]] && binary=fused-align-64
        [[ $mode == a1024 ]] && binary=fused-align-1024
        echo "FUSED_FINAL_FORMAL round=$round mode=$mode m=512 n=19968"
        numactl -N 1 --preferred=17 "./$binary" 512 19968 17
    done
done
sha256sum trsm-fused-align.c
echo FUSED_ALIGN_DONE
