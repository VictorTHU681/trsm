#!/usr/bin/env bash
set -eo pipefail
cd generated
flags=(-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp)
export GLIBC_TUNABLES=glibc.malloc.hugetlb=1
gcc "${flags[@]}" -c trsm-fused-align.c -o aligned-final.o
gcc "${flags[@]}" gen-strict.c aligned-final.o reference.o -o gen-aligned-final -lm
OMP_NUM_THREADS=5 TRSM_FUSED_DIAG=2 TRSM_OWNER=0 TRSM_DIRECT=0 TRSM_MB=192 TRSM_JIT_MB=0 TRSM_PPMODE=1 numactl -N 1 --preferred=17 ./gen-aligned-final 30
gcc "${flags[@]}" independent-fused-close.c aligned-final.o -o aligned-independent -lm
OMP_NUM_THREADS=5 TRSM_OWNER=0 TRSM_ANT=1 TRSM_DIRECT=0 TRSM_MB=192 TRSM_JIT_MB=0 TRSM_PPMODE=1 numactl -N 1 --preferred=17 ./aligned-independent
gcc "${flags[@]}" validate.c aligned-final.o -o validate-aligned -lm -lkblas
numactl -N 1 --preferred=17 ./validate-aligned
mkdir -p aligned-final-run
cp trsm-fused-align.c aligned-final-run/trsm.c
cp bench_trsm.c run.sh aligned-final-run/
cd aligned-final-run
unset TRSM_FUSED_DIAG
bash run.sh
gcc -O3 -march=armv9-a+sme+sme-f64f64+sve2 bench_trsm.c trsm.c -o original-command -lm -lkblas -fopenmp
for shape in '512 19968' '2432 17024' '17024 512'; do
    read -r m n <<<"$shape"
    OMP_NUM_THREADS=38 numactl -N 1 ./original-command "$m" "$n" 1
done
nm -g --defined-only ../aligned-final.o
sha256sum trsm.c bench_trsm.c run.sh
echo FUSED_ALIGN_CLOSE_DONE
