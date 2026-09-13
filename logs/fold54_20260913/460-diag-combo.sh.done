#!/usr/bin/env bash
set -eo pipefail
cd generated
flags=(-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp)
export GLIBC_TUNABLES=glibc.malloc.hugetlb=1
gcc "${flags[@]}" diag-combo-check.c -o diag-combo-check -lm
OMP_NUM_THREADS=1 numactl -N 1 --preferred=17 ./diag-combo-check
gcc "${flags[@]}" bench_trsm.c trsm-diag-combo.c -o diag-combo -lm -lkblas
for shape in '512 19968' '2432 17024' '17024 512'; do
    read -r m n <<<"$shape"
    for round in 1 2 3 4; do
        order=(control off sme div mul)
        [[ $((round%2)) == 0 ]] && order=(mul div sme off control)
        for mode in "${order[@]}"; do
            binary=diag-combo
            export TRSM_DIAG_SME=0 TRSM_DIAG_ROW=0
            case $mode in
                control) binary=control;;
                sme) export TRSM_DIAG_SME=1;;
                div) export TRSM_DIAG_SME=1 TRSM_DIAG_ROW=1;;
                mul) export TRSM_DIAG_SME=1 TRSM_DIAG_ROW=2;;
            esac
            echo "DIAG_COMBO_FORMAL round=$round mode=$mode m=$m n=$n"
            numactl -N 1 --preferred=17 "./$binary" "$m" "$n" 9
        done
    done
    for mode in 0 1 2; do
        echo "DIAG_COMBO_PROFILE mode=$mode m=$m n=$n"
        TRSM_DIAG_SME=1 TRSM_DIAG_ROW=$mode TRSM_PROFILE=3 numactl -N 1 --preferred=17 ./diag-combo "$m" "$n" 5
    done
done
sha256sum trsm-diag-combo.c diag-combo-check.c
