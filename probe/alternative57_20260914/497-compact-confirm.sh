#!/usr/bin/env bash
set -eo pipefail
cd generated
flags=(-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp)
export GLIBC_TUNABLES=glibc.malloc.hugetlb=1
sha256sum trsm-diagcompact.c compact-guard.c independent-diagcompact.c trsm-v55.c bench_trsm.c
gcc "${flags[@]}" compact-guard.c -o compact-guard -lm
OMP_NUM_THREADS=1 numactl -N 1 --preferred=17 ./compact-guard
gcc "${flags[@]}" -DTRSM_SW_RATE=4 compact-guard.c -o compact-guard-rate4 -lm
OMP_NUM_THREADS=1 numactl -N 1 --preferred=17 ./compact-guard-rate4
gcc "${flags[@]}" independent-diagcompact.c -o independent-diagcompact -lm
for on in 0 1; do
    OMP_NUM_THREADS=5 TRSM_DIAG_COMPACT=$on TRSM_OWNER=0 TRSM_ANT=1 TRSM_DIRECT=0 TRSM_MB=256 TRSM_SWHOST=0 TRSM_PPMODE=0 TRSM_JIT_MB=0 TRSM_JIT=0 numactl -N 1 --preferred=17 ./independent-diagcompact
done
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
for mode in 0 1; do
    echo "COMPACT_PROFILE mode=$mode m=2432 n=17024"
    TRSM_DIAG_COMPACT=$mode TRSM_PROFILE=2 numactl -N 1 --preferred=17 ./diagcompact 2432 17024 3
done
echo COMPACT_CONFIRM_DONE
