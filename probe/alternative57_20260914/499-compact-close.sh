#!/usr/bin/env bash
set -eo pipefail
cd generated
flags=(-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp)
export GLIBC_TUNABLES=glibc.malloc.hugetlb=1
unset TRSM_DIAG_COMPACT TRSM_PROFILE
sha256sum trsm-diagcompact.c gen-strict.c validate.c run.sh control diagcompact
gcc "${flags[@]}" -DTRSM_NO_SME_ASM -Dl_trsm=l_trsm_ref -c trsm-v55.c -o reference.o
gcc "${flags[@]}" -c trsm-diagcompact.c -o compact.o
gcc "${flags[@]}" gen-strict.c compact.o reference.o -o gen-compact -lm
echo COMPACT_CLOSE_GENERAL
OMP_NUM_THREADS=5 numactl -N 1 --preferred=17 ./gen-compact 40
echo COMPACT_CLOSE_SPECIAL
OMP_NUM_THREADS=5 TRSM_OWNER=0 TRSM_ANT=1 TRSM_DIRECT=0 TRSM_MB=256 TRSM_SWHOST=0 TRSM_PPMODE=0 TRSM_JIT_MB=0 TRSM_JIT=0 numactl -N 1 --preferred=17 ./gen-compact 10
gcc "${flags[@]}" validate.c compact.o -o validate-compact -lm -lkblas
echo COMPACT_CLOSE_CHANGED_INPUT
numactl -N 1 --preferred=17 ./validate-compact
mkdir -p compact-final-run
cp trsm-diagcompact.c compact-final-run/trsm.c
cp bench_trsm.c run.sh compact-final-run/
cd compact-final-run
echo COMPACT_CLOSE_RUNSH
bash run.sh
echo COMPACT_CLOSE_ORIGINAL
gcc -O3 -march=armv9-a+sme+sme-f64f64+sve2 bench_trsm.c trsm.c -o original-command -lm -lkblas -fopenmp
for shape in '512 19968' '2432 17024' '17024 512'; do
    read -r m n <<<"$shape"
    OMP_NUM_THREADS=38 numactl -N 1 ./original-command "$m" "$n" 1
done
sha256sum trsm.c bench_trsm.c run.sh
echo COMPACT_CLOSE_DONE
