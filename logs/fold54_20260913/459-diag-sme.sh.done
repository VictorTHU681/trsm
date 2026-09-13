#!/usr/bin/env bash
set -eo pipefail
cd generated
flags=(-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp)
export GLIBC_TUNABLES=glibc.malloc.hugetlb=1
gcc "${flags[@]}" diag-sme-check.c -o diag-sme-check -lm
OMP_NUM_THREADS=1 numactl -N 1 --preferred=17 ./diag-sme-check
gcc "${flags[@]}" independent-fold.c trsm-diag-sme.c -o independent-diag-sme-owner -lm
OMP_NUM_THREADS=5 TRSM_DIAG_SME=1 numactl -N 1 --preferred=17 ./independent-diag-sme-owner
gcc "${flags[@]}" independent-duplex.c trsm-diag-sme.c -o independent-diag-sme-duplex -lm
OMP_NUM_THREADS=5 TRSM_DIAG_SME=1 TRSM_OWNER=0 TRSM_ANT=1 TRSM_DIRECT=0 TRSM_MB=256 TRSM_SWHOST=0 numactl -N 1 --preferred=17 ./independent-diag-sme-duplex
OMP_NUM_THREADS=5 TRSM_DIAG_SME=1 TRSM_OWNER=0 TRSM_ANT=1 TRSM_DIRECT=0 TRSM_MB=192 TRSM_JIT_MB=0 TRSM_PPMODE=1 numactl -N 1 --preferred=17 ./independent-diag-sme-duplex
gcc "${flags[@]}" bench_trsm.c trsm-diag-sme.c -o diag-sme -lm -lkblas
for shape in '512 19968' '2432 17024' '17024 512'; do
    read -r m n <<<"$shape"
    for round in 1 2 3 4; do
        order=(control off on)
        [[ $((round%2)) == 0 ]] && order=(on off control)
        for mode in "${order[@]}"; do
            binary=diag-sme
            export TRSM_DIAG_SME=0
            [[ $mode == control ]] && binary=control
            [[ $mode == on ]] && export TRSM_DIAG_SME=1
            echo "DIAG_SME_FORMAL round=$round mode=$mode m=$m n=$n"
            numactl -N 1 --preferred=17 "./$binary" "$m" "$n" 9
        done
    done
    for mode in 0 1; do
        echo "DIAG_SME_PROFILE mode=$mode m=$m n=$n"
        TRSM_DIAG_SME=$mode TRSM_PROFILE=3 numactl -N 1 --preferred=17 ./diag-sme "$m" "$n" 5
    done
done
sha256sum trsm-diag-sme.c diag-sme-check.c
