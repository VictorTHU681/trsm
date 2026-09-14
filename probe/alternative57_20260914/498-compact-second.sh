#!/usr/bin/env bash
set -eo pipefail
cd generated
flags=(-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp)
export GLIBC_TUNABLES=glibc.malloc.hugetlb=1
sha256sum trsm-diagcompact.c trsm-v55.c bench_trsm.c
gcc "${flags[@]}" bench_trsm.c trsm-v55.c -o control -lm -lkblas
gcc "${flags[@]}" bench_trsm.c trsm-diagcompact.c -o diagcompact -lm -lkblas
orders=('control on off' 'off control on' 'on off control' 'control off on' 'on control off' 'off on control')
for shape in '512 19968' '2432 17024' '17024 512'; do
    read -r m n <<<"$shape"
    for round in $(seq 1 12); do
        read -ra order <<<"${orders[$(((round-1)%6))]}"
        for mode in "${order[@]}"; do
            binary=diagcompact
            export TRSM_DIAG_COMPACT=1
            case "$mode" in control) binary=control;; off) export TRSM_DIAG_COMPACT=0;; esac
            echo "COMPACT_FORMAL round=$round mode=$mode m=$m n=$n"
            numactl -N 1 --preferred=17 "./$binary" "$m" "$n" 11
        done
    done
done
sha256sum control diagcompact
echo COMPACT_SECOND_DONE
touch ../STOP
