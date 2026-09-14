#!/usr/bin/env bash
set -eo pipefail
cd generated
flags=(-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp)
export GLIBC_TUNABLES=glibc.malloc.hugetlb=1
unset TRSM_DIAG_COMPACT TRSM_PROFILE
sha256sum trsm-v57.c compact-guard.c independent-v57.c gen-strict.c validate.c run.sh
gcc "${flags[@]}" compact-guard.c -o compact-guard -lm
numactl -N 1 --preferred=17 ./compact-guard
gcc "${flags[@]}" independent-v57.c -o independent-v57 -lm
for mode in 0 1; do
    OMP_NUM_THREADS=5 TRSM_DIAG_COMPACT=$mode TRSM_OWNER=0 TRSM_ANT=1 TRSM_DIRECT=0 TRSM_MB=192 TRSM_JIT_MB=0 TRSM_PPMODE=1 TRSM_JIT=0 numactl -N 1 --preferred=17 ./independent-v57
done
gcc "${flags[@]}" -DTRSM_NO_SME_ASM -Dl_trsm=l_trsm_ref -c trsm-v56.c -o reference.o
gcc "${flags[@]}" -c trsm-v57.c -o candidate.o
gcc "${flags[@]}" gen-strict.c candidate.o reference.o -o gen-candidate -lm
echo FINAL58_GENERAL
OMP_NUM_THREADS=5 numactl -N 1 --preferred=17 ./gen-candidate 40
echo FINAL58_SPECIAL4
OMP_NUM_THREADS=5 TRSM_OWNER=0 TRSM_ANT=1 TRSM_DIRECT=0 TRSM_MB=192 TRSM_SWHOST=0 TRSM_PPMODE=1 TRSM_JIT_MB=0 TRSM_JIT=0 numactl -N 1 --preferred=17 ./gen-candidate 10
gcc "${flags[@]}" validate.c candidate.o -o validate-candidate -lm -lkblas
echo FINAL58_CHANGED_INPUT
numactl -N 1 --preferred=17 ./validate-candidate
mkdir -p final-run
cp trsm-v57.c final-run/trsm.c
cp bench_trsm.c run.sh final-run/
cd final-run
echo FINAL58_RUNSH
bash run.sh
echo FINAL58_ORIGINAL
gcc -O3 -march=armv9-a+sme+sme-f64f64+sve2 bench_trsm.c trsm.c -o original-command -lm -lkblas -fopenmp
for shape in '512 19968' '2432 17024' '17024 512'; do
    read -r m n <<<"$shape"
    OMP_NUM_THREADS=38 numactl -N 1 ./original-command "$m" "$n" 1
done
sha256sum trsm.c bench_trsm.c run.sh
echo FINAL58_VALIDATE_DONE
