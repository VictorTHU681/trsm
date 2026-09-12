#!/usr/bin/env bash
set -eo pipefail
cd generated
flags=(-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp)
export GLIBC_TUNABLES=glibc.malloc.hugetlb=1
gcc "${flags[@]}" bench_trsm.c fullinv-v3.c -o fullinv-v3 -lm -lkblas
for round in 1 2; do
    for config in '0 0' '1 0' '2 0' '0 1' '1 1' '2 1'; do
        read -r kernel pack <<<"$config"
        echo "INVERSE_COMPACT round=$round kernel=$kernel pack=$pack nc=128"
        TRSM_FULLINV_KERNEL=$kernel TRSM_FULLINV_PACK=$pack TRSM_FULLINV_NC=128 TRSM_FULLINV_PROFILE=1 numactl -N 1 --preferred=17 ./fullinv-v3 512 19968 5 | tee "inv-compact-$round-$kernel-$pack.log"
        grep -q 'PASS' "inv-compact-$round-$kernel-$pack.log"
        if grep -q 'FAIL' "inv-compact-$round-$kernel-$pack.log"; then exit 3; fi
    done
done
