#!/usr/bin/env bash
set -eo pipefail
cd generated
flags=(-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp)
export GLIBC_TUNABLES=glibc.malloc.hugetlb=1
gcc "${flags[@]}" independent-fused-close.c fused-audit.o -o independent-fused-close -lm
OMP_NUM_THREADS=5 TRSM_OWNER=0 TRSM_ANT=1 TRSM_DIRECT=0 TRSM_MB=192 TRSM_JIT_MB=0 TRSM_PPMODE=1 numactl -N 1 --preferred=17 ./independent-fused-close
for knobs in 'TRSM_NOBAR=0' 'TRSM_PPMODE=0' 'TRSM_TBUILD=0 TRSM_PACK_LA=0' 'TRSM_TBUILD=1 TRSM_PACK_LA=1'; do
    echo "FUSED_SETTINGS $knobs"
    env OMP_NUM_THREADS=5 TRSM_OWNER=0 TRSM_DIRECT=0 TRSM_MB=192 TRSM_JIT_MB=0 TRSM_PPMODE=1 TRSM_FUSED_DIAG=2 $knobs numactl -N 1 --preferred=17 ./gen-fused 10
done
mkdir -p fused-final-run
cp trsm-fused-final.c fused-final-run/trsm.c
cp bench_trsm.c run.sh fused-final-run/
cd fused-final-run
unset TRSM_FUSED_DIAG
bash run.sh
gcc -O3 -march=armv9-a+sme+sme-f64f64+sve2 bench_trsm.c trsm.c -o original-command -lm -lkblas -fopenmp
for shape in '512 19968' '2432 17024' '17024 512'; do
    read -r m n <<<"$shape"
    OMP_NUM_THREADS=38 numactl -N 1 ./original-command "$m" "$n" 1
done
nm -g --defined-only ../fused-final.o
sha256sum trsm.c bench_trsm.c run.sh
echo FUSED_FINAL_CLOSE_DONE
