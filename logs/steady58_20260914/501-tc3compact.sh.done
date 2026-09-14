#!/usr/bin/env bash
set -eo pipefail
cd generated
flags=(-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp)
export GLIBC_TUNABLES=glibc.malloc.hugetlb=1
sha256sum trsm-tc3compact.c independent-tc3compact.c
gcc "${flags[@]}" independent-tc3compact.c -o independent-tc3compact -lm
for mode in 1 2; do
    OMP_NUM_THREADS=5 TRSM_DIAG_COMPACT=$mode TRSM_OWNER=0 TRSM_ANT=1 TRSM_DIRECT=0 TRSM_MB=192 TRSM_JIT_MB=0 TRSM_PPMODE=1 TRSM_JIT=0 numactl -N 1 --preferred=17 ./independent-tc3compact
done
gcc "${flags[@]}" bench_trsm.c trsm-tc3compact.c -o tc3compact -lm -lkblas
orders=('control on off' 'off control on' 'on off control' 'control off on' 'on control off' 'off on control')
for round in $(seq 1 12); do
    read -ra order <<<"${orders[$(((round-1)%6))]}"
    for mode in "${order[@]}"; do
        binary=tc3compact
        export TRSM_DIAG_COMPACT=1
        case "$mode" in control) binary=control;; on) export TRSM_DIAG_COMPACT=2;; esac
        echo "TC3COMPACT_FORMAL round=$round mode=$mode m=17024 n=512"
        numactl -N 1 --preferred=17 "./$binary" 17024 512 11
    done
done
for mode in 1 2; do
    echo "TC3COMPACT_PROFILE mode=$mode m=17024 n=512"
    TRSM_DIAG_COMPACT=$mode TRSM_PROFILE=2 numactl -N 1 --preferred=17 ./tc3compact 17024 512 3
done
echo TC3COMPACT_DONE
