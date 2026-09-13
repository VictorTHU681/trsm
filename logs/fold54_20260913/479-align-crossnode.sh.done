#!/usr/bin/env bash
set -eo pipefail
cd /home/share/niujiayin/trsm-diag55-20260913/generated
export GLIBC_TUNABLES=glibc.malloc.hugetlb=1
orders=('control final off' 'off control final' 'final off control' 'control off final' 'final control off' 'off final control')
for shape in '512 19968' '2432 17024' '17024 512'; do
    read -r m n <<<"$shape"
    for round in 1 2 3 4 5 6 7 8 9 10 11 12; do
        read -ra order <<<"${orders[$(((round-1)%6))]}"
        for mode in "${order[@]}"; do
            binary=fused-align-64
            unset TRSM_FUSED_DIAG
            [[ $mode == control ]] && binary=control
            [[ $mode == off ]] && export TRSM_FUSED_DIAG=0
            echo "FUSED_FINAL_FORMAL round=$round mode=$mode m=$m n=$n"
            numactl -N 1 --preferred=17 "./$binary" "$m" "$n" 17
        done
    done
done
sha256sum trsm-fused-align.c fused-align-64 control
echo FUSED_ALIGN_CROSS_DONE
