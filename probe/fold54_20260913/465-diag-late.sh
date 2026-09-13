#!/usr/bin/env bash
set -eo pipefail
cd generated
flags=(-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp)
export GLIBC_TUNABLES=glibc.malloc.hugetlb=1
gcc "${flags[@]}" -c trsm-diag-late.c -o diag-late.o
gcc "${flags[@]}" independent-duplex.c diag-late.o -o independent-diag-late -lm
OMP_NUM_THREADS=5 TRSM_DIAG_SME=1 TRSM_LATE_DIAG=1 TRSM_EXPECT_LATE=1 TRSM_OWNER=0 TRSM_ANT=1 TRSM_DIRECT=0 TRSM_MB=192 TRSM_JIT_MB=0 TRSM_PPMODE=1 numactl -N 1 --preferred=17 ./independent-diag-late
gcc "${flags[@]}" gen-strict.c diag-late.o reference.o -o gen-diag-late -lm
OMP_NUM_THREADS=5 TRSM_DIAG_SME=1 TRSM_LATE_DIAG=1 TRSM_EXPECT_LATE=1 TRSM_OWNER=0 TRSM_DIRECT=0 TRSM_MB=192 TRSM_JIT_MB=0 TRSM_PPMODE=1 numactl -N 1 --preferred=17 ./gen-diag-late 30
gcc "${flags[@]}" bench_trsm.c diag-late.o -o diag-late -lm -lkblas
for round in 1 2 3 4 5 6; do
    order=(control off sme late_old late_sme)
    [[ $((round%2)) == 0 ]] && order=(late_sme late_old sme off control)
    for mode in "${order[@]}"; do
        binary=diag-late
        export TRSM_DIAG_SME=0 TRSM_LATE_DIAG=0
        case $mode in
            control) binary=control;;
            sme) export TRSM_DIAG_SME=1;;
            late_old) export TRSM_LATE_DIAG=1;;
            late_sme) export TRSM_LATE_DIAG=1 TRSM_DIAG_SME=1;;
        esac
        echo "DIAG_LATE_FORMAL round=$round mode=$mode m=17024 n=512"
        numactl -N 1 --preferred=17 "./$binary" 17024 512 13
    done
done
for mode in off sme late_old late_sme; do
    export TRSM_DIAG_SME=0 TRSM_LATE_DIAG=0
    [[ $mode == sme || $mode == late_sme ]] && export TRSM_DIAG_SME=1
    [[ $mode == late_old || $mode == late_sme ]] && export TRSM_LATE_DIAG=1
    echo "DIAG_LATE_PROFILE mode=$mode m=17024 n=512"
    TRSM_PROFILE=3 numactl -N 1 --preferred=17 ./diag-late 17024 512 5
done
sha256sum trsm-diag-late.c
