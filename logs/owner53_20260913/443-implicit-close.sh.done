#!/usr/bin/env bash
set -eo pipefail
cd generated
flags=(-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp)
export GLIBC_TUNABLES=glibc.malloc.hugetlb=1
gcc "${flags[@]}" final-bounds.c -o final-bounds -lm
OMP_NUM_THREADS=1 numactl -N 1 --preferred=17 ./final-bounds
gcc "${flags[@]}" implicit-gate.c implicit-final-audit.o v53-ref.o -o implicit-gate -lm
OMP_NUM_THREADS=1 numactl -N 1 --preferred=17 ./implicit-gate
OMP_NUM_THREADS=38 numactl -N 1 --preferred=17 ./implicit-gate
gcc "${flags[@]}" implicit-input-audit.c trsm-implicit-close.c -o implicit-input-audit -lm -lkblas
numactl -N 1 --preferred=17 ./implicit-input-audit
for round in 1 2 3 4 5 6 7 8 9 10 11 12; do
    order=(control final off)
    [[ $((round%2)) == 0 ]] && order=(off final control)
    for mode in "${order[@]}"; do
        unset TRSM_IMPLICIT_OWNER
        binary=implicit-final
        [[ $mode == control ]] && binary=control
        [[ $mode == off ]] && export TRSM_IMPLICIT_OWNER=0
        echo "IMPLICIT_FINAL_FORMAL round=$round mode=$mode m=512 n=19968"
        numactl -N 1 --preferred=17 "./$binary" 512 19968 11
    done
done
unset TRSM_IMPLICIT_OWNER
for mode in control final off; do
    binary=implicit-final
    [[ $mode == control ]] && binary=control
    [[ $mode == off ]] && export TRSM_IMPLICIT_OWNER=0
    echo "IMPLICIT_FINAL_PROFILE mode=$mode m=512 n=19968"
    TRSM_PROFILE=3 numactl -N 1 --preferred=17 "./$binary" 512 19968 7
done
unset TRSM_IMPLICIT_OWNER
mkdir -p submission-check
cp trsm-implicit-final.c submission-check/trsm.c
cp bench_trsm.c run.sh submission-check/
echo IMPLICIT_RUN_SH
bash submission-check/run.sh
echo IMPLICIT_ORIGINAL_COMMANDS
for shape in '512 19968' '2432 17024' '17024 512'; do
    numactl -N 1 ./submission-check/trsm_test $shape 1
done
gcc "${flags[@]}" -c trsm-implicit-final.c -o implicit-final-production.o
nm --defined-only --extern-only implicit-final-production.o
sha256sum trsm-implicit-final.c bench_trsm.c run.sh submission-check/trsm.c
echo IMPLICIT_CLOSE_DONE
