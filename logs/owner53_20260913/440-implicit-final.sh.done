#!/usr/bin/env bash
set -eo pipefail
cd generated
flags=(-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp)
export GLIBC_TUNABLES=glibc.malloc.hugetlb=1
gcc "${flags[@]}" bench_trsm.c trsm-implicit-final.c -o implicit-final -lm -lkblas
gcc "${flags[@]}" -c trsm-implicit-final-audit.c -o implicit-final-audit.o
gcc -O3 -fopenmp -DTRSM_NO_SME_ASM -Dl_trsm=l_trsm_ref -c trsm-v53.c -o v53-ref.o
gcc "${flags[@]}" gen-strict.c implicit-final-audit.o v53-ref.o -o gen-final -lm
numactl -N 1 --preferred=17 ./gen-final 30
OMP_NUM_THREADS=5 TRSM_OWNER=1 numactl -N 1 --preferred=17 ./gen-final 20
gcc "${flags[@]}" independent-owner53.c implicit-final-audit.o -o independent-final -lm
OMP_NUM_THREADS=5 numactl -N 1 --preferred=17 ./independent-final
gcc "${flags[@]}" independent-duplex.c implicit-final-audit.o -o independent-duplex-final -lm
OMP_NUM_THREADS=5 TRSM_OWNER=0 TRSM_ANT=1 TRSM_DIRECT=0 TRSM_MB=256 TRSM_SWHOST=0 TRSM_EXPECT_IMPLICIT=0 numactl -N 1 --preferred=17 ./independent-duplex-final
OMP_NUM_THREADS=5 TRSM_OWNER=0 TRSM_ANT=1 TRSM_DIRECT=0 TRSM_MB=192 TRSM_JIT_MB=0 TRSM_PPMODE=1 TRSM_EXPECT_IMPLICIT=0 numactl -N 1 --preferred=17 ./independent-duplex-final
gcc "${flags[@]}" validate.c implicit-final-audit.o -o validate-final -lm -lkblas
numactl -N 1 --preferred=17 ./validate-final
echo IMPLICIT_FINAL_VALIDATION_DONE
for round in 1 2 3 4 5 6; do
    order=(control final off)
    [[ $((round%2)) == 0 ]] && order=(off final control)
    for shape in '512 19968' '2432 17024' '17024 512'; do
        read -r m n <<<"$shape"
        for mode in "${order[@]}"; do
            unset TRSM_IMPLICIT_OWNER
            binary=implicit-final
            [[ $mode == control ]] && binary=control
            [[ $mode == off ]] && export TRSM_IMPLICIT_OWNER=0
            echo "IMPLICIT_FINAL_FORMAL round=$round mode=$mode m=$m n=$n"
            numactl -N 1 --preferred=17 "./$binary" "$m" "$n" 11
        done
    done
done
