#!/usr/bin/env bash
set -eo pipefail
cd generated
flags=(-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp)
export GLIBC_TUNABLES=glibc.malloc.hugetlb=1
gcc "${flags[@]}" independent-duplex.c trsm-diagstore.c -o independent-diagstore -lm
OMP_NUM_THREADS=5 TRSM_DIAG_STORE=1 TRSM_DIAG_STORE_MAX=0 TRSM_OWNER=0 TRSM_ANT=1 TRSM_DIRECT=0 TRSM_MB=256 TRSM_SWHOST=0 numactl -N 1 --preferred=17 ./independent-diagstore
OMP_NUM_THREADS=5 TRSM_DIAG_STORE=2 TRSM_DIAG_STORE_MAX=64 TRSM_OWNER=0 TRSM_ANT=1 TRSM_DIRECT=0 TRSM_MB=192 TRSM_JIT_MB=0 TRSM_PPMODE=1 numactl -N 1 --preferred=17 ./independent-diagstore
gcc "${flags[@]}" bench_trsm.c trsm-diagstore.c -o diagstore -lm -lkblas
for shape in '2432 17024' '17024 512'; do
    read -r m n <<<"$shape"
    for round in 1 2 3; do
        order=(control off za64 mova64 zaall movaall)
        [[ $((round%2)) == 0 ]] && order=(movaall zaall mova64 za64 off control)
        for mode in "${order[@]}"; do
            binary=diagstore
            export TRSM_DIAG_STORE=0 TRSM_DIAG_STORE_MAX=64
            case "$mode" in
                control) binary=control;;
                za64) export TRSM_DIAG_STORE=1;;
                mova64) export TRSM_DIAG_STORE=2;;
                zaall) export TRSM_DIAG_STORE=1 TRSM_DIAG_STORE_MAX=0;;
                movaall) export TRSM_DIAG_STORE=2 TRSM_DIAG_STORE_MAX=0;;
            esac
            echo "DIAGSTORE_FORMAL round=$round mode=$mode m=$m n=$n"
            numactl -N 1 --preferred=17 "./$binary" "$m" "$n" 7
        done
    done
done
for mode in off mova64 movaall; do
    export TRSM_DIAG_STORE=0 TRSM_DIAG_STORE_MAX=64
    [[ $mode == mova64 ]] && export TRSM_DIAG_STORE=2
    [[ $mode == movaall ]] && export TRSM_DIAG_STORE=2 TRSM_DIAG_STORE_MAX=0
    echo "DIAGSTORE_PROFILE mode=$mode m=2432 n=17024"
    TRSM_PROFILE=3 numactl -N 1 --preferred=17 ./diagstore 2432 17024 5
done
sha256sum trsm-diagstore.c
