#!/usr/bin/env bash
set -eo pipefail
cd generated
flags=(-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp)
export GLIBC_TUNABLES=glibc.malloc.hugetlb=1
gcc "${flags[@]}" -c trsm-diag-close-audit.c -o diag-close-audit.o
gcc "${flags[@]}" diag-final-input-audit.c diag-close-audit.o -o diag-close-input -lm -lkblas
OMP_NUM_THREADS=5 numactl -N 1 --preferred=17 ./diag-close-input
gcc "${flags[@]}" independent-diag-extreme-fold.c diag-close-audit.o -o diag-extreme-owner -lm
OMP_NUM_THREADS=5 numactl -N 1 --preferred=17 ./diag-extreme-owner
gcc "${flags[@]}" independent-diag-extreme-duplex.c diag-close-audit.o -o diag-extreme-duplex -lm
OMP_NUM_THREADS=5 TRSM_OWNER=0 TRSM_ANT=1 TRSM_DIRECT=0 TRSM_MB=256 TRSM_SWHOST=0 numactl -N 1 --preferred=17 ./diag-extreme-duplex
OMP_NUM_THREADS=5 TRSM_OWNER=0 TRSM_ANT=1 TRSM_DIRECT=0 TRSM_MB=192 TRSM_JIT_MB=0 TRSM_PPMODE=1 numactl -N 1 --preferred=17 ./diag-extreme-duplex
gcc "${flags[@]}" diag-gate.c diag-close-audit.o reference.o -o diag-gate -lm
OMP_NUM_THREADS=5 numactl -N 1 --preferred=17 ./diag-gate
mkdir -p final-run
cp trsm-diag-final.c final-run/trsm.c
cp bench_trsm.c run.sh final-run/
cd final-run
unset TRSM_DIAG_SME
bash run.sh
gcc -O3 -march=armv9-a+sme+sme-f64f64+sve2 bench_trsm.c trsm.c -o original-command -lm -lkblas -fopenmp
for shape in '512 19968' '2432 17024' '17024 512'; do
    read -r m n <<<"$shape"
    OMP_NUM_THREADS=38 numactl -N 1 ./original-command "$m" "$n" 1
done
nm -g --defined-only ../diag-final.o
sha256sum trsm.c bench_trsm.c run.sh
echo DIAG_FINAL_CLOSE_DONE
