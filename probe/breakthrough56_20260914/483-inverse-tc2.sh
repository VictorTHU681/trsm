#!/usr/bin/env bash
set -eo pipefail
cd generated
flags=(-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp)
export GLIBC_TUNABLES=glibc.malloc.hugetlb=1
gcc "${flags[@]}" bench_trsm.c trsm-v55.c -o control -lm -lkblas
gcc "${flags[@]}" bench_trsm.c fullinv-v3.c -o fullinv -lm -lkblas
gcc "${flags[@]}" independent-fullinv.c fullinv-v3-audit.c -o inverse-known -lm
OMP_NUM_THREADS=5 numactl -N 1 --preferred=17 ./inverse-known
for shape in '512 19968' '2432 17024'; do
    read -r m n <<<"$shape"
    for round in 1 2 3; do
        for mode in control off inv32 inv128; do
            binary=fullinv
            export TRSM_FULLINV=1 TRSM_FULLINV_NC=128 TRSM_FULLINV_PACK=0
            [[ $mode == control ]] && binary=control
            [[ $mode == off ]] && export TRSM_FULLINV=0
            [[ $mode == inv32 ]] && export TRSM_FULLINV_NC=32
            echo "INV56_FORMAL round=$round mode=$mode m=$m n=$n"
            numactl -N 1 --preferred=17 "./$binary" "$m" "$n" 7
        done
    done
    echo "INV56_PROFILE m=$m n=$n"
    TRSM_FULLINV_PROFILE=1 numactl -N 1 --preferred=17 ./fullinv "$m" "$n" 3
done
sha256sum trsm-v55.c fullinv-v3.c fullinv-compact.h bench_trsm.c
echo INVERSE_TC2_DONE
