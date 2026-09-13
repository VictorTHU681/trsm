#!/usr/bin/env bash
set -eo pipefail
cd generated
flags=(-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp)
export GLIBC_TUNABLES=glibc.malloc.hugetlb=1
gcc "${flags[@]}" -c trsm-diag-final.c -o diag-final.o
gcc "${flags[@]}" bench_trsm.c diag-final.o -o diag-final -lm -lkblas
gcc "${flags[@]}" bench_trsm.c trsm-v54.c -o control -lm -lkblas
gcc -O3 -fopenmp -DTRSM_NO_SME_ASM -Dl_trsm=l_trsm_ref -c trsm-v54.c -o reference.o
gcc "${flags[@]}" gen-strict.c diag-final.o reference.o -o gen-diag-final -lm
TRSM_DIAG_SME=1 numactl -N 1 --preferred=17 ./gen-diag-final 30
OMP_NUM_THREADS=5 TRSM_DIAG_SME=1 TRSM_OWNER=1 numactl -N 1 --preferred=17 ./gen-diag-final 20
gcc "${flags[@]}" independent-fold.c diag-final.o -o independent-diag-final-owner -lm
OMP_NUM_THREADS=5 TRSM_DIAG_SME=1 numactl -N 1 --preferred=17 ./independent-diag-final-owner
gcc "${flags[@]}" independent-duplex.c diag-final.o -o independent-diag-final-duplex -lm
OMP_NUM_THREADS=5 TRSM_DIAG_SME=1 TRSM_OWNER=0 TRSM_ANT=1 TRSM_DIRECT=0 TRSM_MB=256 TRSM_SWHOST=0 numactl -N 1 --preferred=17 ./independent-diag-final-duplex
OMP_NUM_THREADS=5 TRSM_DIAG_SME=1 TRSM_OWNER=0 TRSM_ANT=1 TRSM_DIRECT=0 TRSM_MB=192 TRSM_JIT_MB=0 TRSM_PPMODE=1 numactl -N 1 --preferred=17 ./independent-diag-final-duplex
gcc "${flags[@]}" validate.c diag-final.o -o validate-diag-final -lm -lkblas
numactl -N 1 --preferred=17 ./validate-diag-final
gcc "${flags[@]}" diag-final-check.c -o diag-final-check -lm
OMP_NUM_THREADS=1 numactl -N 1 --preferred=17 ./diag-final-check
gcc "${flags[@]}" diag-final-input-audit.c trsm-diag-final-audit.c -o diag-final-input-audit -lm -lkblas
OMP_NUM_THREADS=5 TRSM_DIAG_SME=1 numactl -N 1 --preferred=17 ./diag-final-input-audit
echo DIAG_FINAL_VALIDATION_DONE
for shape in '512 19968' '2432 17024' '17024 512'; do
    read -r m n <<<"$shape"
    for round in 1 2 3 4 5 6 7 8 9 10 11 12; do
        order=(control final off)
        [[ $((round%2)) == 0 ]] && order=(off final control)
        for mode in "${order[@]}"; do
            binary=diag-final
            unset TRSM_DIAG_SME
            [[ $mode == control ]] && binary=control
            [[ $mode == off ]] && export TRSM_DIAG_SME=0
            echo "DIAG_FINAL_FORMAL round=$round mode=$mode m=$m n=$n"
            numactl -N 1 --preferred=17 "./$binary" "$m" "$n" 13
        done
    done
done
sha256sum trsm-diag-final.c
echo DIAG_FINAL_SCREEN_DONE
