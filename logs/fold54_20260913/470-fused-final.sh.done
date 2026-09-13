#!/usr/bin/env bash
set -eo pipefail
cd /home/share/niujiayin/trsm-diag55-20260913/generated
flags=(-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp)
export GLIBC_TUNABLES=glibc.malloc.hugetlb=1
gcc "${flags[@]}" -c trsm-fused-final.c -o fused-final.o
gcc "${flags[@]}" -c trsm-fused-audit.c -o fused-audit.o
gcc "${flags[@]}" gen-strict.c fused-audit.o reference.o -o gen-fused -lm
OMP_NUM_THREADS=5 TRSM_FUSED_DIAG=2 TRSM_OWNER=0 TRSM_DIRECT=0 TRSM_MB=192 TRSM_JIT_MB=0 TRSM_PPMODE=1 numactl -N 1 --preferred=17 ./gen-fused 30
gcc "${flags[@]}" independent-fused.c fused-audit.o -o independent-fused -lm
OMP_NUM_THREADS=5 TRSM_OWNER=0 TRSM_ANT=1 TRSM_DIRECT=0 TRSM_MB=192 TRSM_JIT_MB=0 TRSM_PPMODE=1 numactl -N 1 --preferred=17 ./independent-fused
gcc "${flags[@]}" fused-gate.c fused-audit.o reference.o -o fused-gate -lm
OMP_NUM_THREADS=5 TRSM_JIT_MB=0 numactl -N 1 --preferred=17 ./fused-gate
gcc "${flags[@]}" validate.c fused-audit.o -o validate-fused -lm -lkblas
numactl -N 1 --preferred=17 ./validate-fused
gcc "${flags[@]}" fused-input-audit.c fused-audit.o -o fused-input -lm -lkblas
OMP_NUM_THREADS=5 numactl -N 1 --preferred=17 ./fused-input
gcc "${flags[@]}" fused-check.c -o fused-check -lm
OMP_NUM_THREADS=1 numactl -N 1 --preferred=17 ./fused-check
echo FUSED_FINAL_VALIDATION_DONE
gcc "${flags[@]}" bench_trsm.c fused-final.o -o fused-final -lm -lkblas
for shape in '512 19968' '2432 17024' '17024 512'; do
    read -r m n <<<"$shape"
    for round in 1 2 3 4 5 6 7 8 9 10 11 12; do
        order=(control final off)
        [[ $((round%2)) == 0 ]] && order=(off final control)
        for mode in "${order[@]}"; do
            binary=fused-final
            unset TRSM_FUSED_DIAG
            [[ $mode == control ]] && binary=control
            [[ $mode == off ]] && export TRSM_FUSED_DIAG=0
            echo "FUSED_FINAL_FORMAL round=$round mode=$mode m=$m n=$n"
            numactl -N 1 --preferred=17 "./$binary" "$m" "$n" 13
        done
    done
done
sha256sum trsm-fused-final.c
echo FUSED_FINAL_SCREEN_DONE
