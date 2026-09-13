#!/usr/bin/env bash
set -eo pipefail
cd generated
flags=(-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp)
export GLIBC_TUNABLES=glibc.malloc.hugetlb=1
gcc "${flags[@]}" -c trsm-diag-fuse.c -o diag-fuse.o
gcc "${flags[@]}" independent-duplex.c diag-fuse.o -o independent-diag-fuse -lm
OMP_NUM_THREADS=5 TRSM_DIAG_SME=1 TRSM_LATE_DIAG=2 TRSM_EXPECT_LATE=2 TRSM_OWNER=0 TRSM_ANT=1 TRSM_DIRECT=0 TRSM_MB=192 TRSM_JIT_MB=0 TRSM_PPMODE=1 numactl -N 1 --preferred=17 ./independent-diag-fuse
gcc "${flags[@]}" gen-strict.c diag-fuse.o reference.o -o gen-diag-fuse -lm
OMP_NUM_THREADS=5 TRSM_DIAG_SME=1 TRSM_LATE_DIAG=2 TRSM_EXPECT_LATE=2 TRSM_OWNER=0 TRSM_DIRECT=0 TRSM_MB=192 TRSM_JIT_MB=0 TRSM_PPMODE=1 numactl -N 1 --preferred=17 ./gen-diag-fuse 30
gcc "${flags[@]}" bench_trsm.c diag-fuse.o -o diag-fuse -lm -lkblas
for round in 1 2 3 4 5 6; do
    order=(control off sme late fused_old fused_sme)
    [[ $((round%2)) == 0 ]] && order=(fused_sme fused_old late sme off control)
    for mode in "${order[@]}"; do
        binary=diag-fuse
        export TRSM_DIAG_SME=0 TRSM_LATE_DIAG=0
        case $mode in
            control) binary=control;;
            sme) export TRSM_DIAG_SME=1;;
            late) export TRSM_DIAG_SME=1 TRSM_LATE_DIAG=1;;
            fused_old) export TRSM_LATE_DIAG=2;;
            fused_sme) export TRSM_LATE_DIAG=2 TRSM_DIAG_SME=1;;
        esac
        echo "DIAG_FUSE_FORMAL round=$round mode=$mode m=17024 n=512"
        numactl -N 1 --preferred=17 "./$binary" 17024 512 13
    done
done
for mode in off sme fused_old fused_sme; do
    export TRSM_DIAG_SME=0 TRSM_LATE_DIAG=0
    [[ $mode == sme || $mode == fused_sme ]] && export TRSM_DIAG_SME=1
    [[ $mode == fused_old || $mode == fused_sme ]] && export TRSM_LATE_DIAG=2
    echo "DIAG_FUSE_PROFILE mode=$mode m=17024 n=512"
    TRSM_PROFILE=3 numactl -N 1 --preferred=17 ./diag-fuse 17024 512 5
done
sha256sum trsm-diag-fuse.c
