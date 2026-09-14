#!/usr/bin/env bash
set -eo pipefail
cd generated
export GLIBC_TUNABLES=glibc.malloc.hugetlb=1
gcc -O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp bench_trsm.c trsm-v56.c -o control -lm -lkblas
for shape in '512 19968' '2432 17024' '17024 512'; do
    read -r m n <<<"$shape"
    echo "STEADY_PROFILE m=$m n=$n"
    TRSM_PROFILE=2 numactl -N 1 --preferred=17 ./control "$m" "$n" 3
done
nm -S control | grep -E 'run_task_(hoist[24]|owner_cursor)|kloop_plain_nt'
sha256sum trsm-v56.c bench_trsm.c
echo STEADY_PROFILE_DONE
