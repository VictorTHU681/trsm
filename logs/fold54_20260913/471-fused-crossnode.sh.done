#!/usr/bin/env bash
set -eo pipefail
cd generated
export GLIBC_TUNABLES=glibc.malloc.hugetlb=1
sha256sum trsm-fused-final.c fused-final control
for shape in '512 19968' '2432 17024' '17024 512'; do
    read -r m n <<<"$shape"
    for round in 1 2 3 4 5 6 7 8 9 10 11 12; do
        order=(control final off)
        [[ $((round%2)) == 0 ]] && order=(off final control)
        for mode in "${order[@]}"; do
            binary=fused-final
            unset TRSM_FUSED_DIAG
            [[ $mode == control ]] && binary=control
            [[ $mode == off ]] && export TRSM_FUSED_DIAG=0
            echo "FUSED_FINAL_FORMAL round=$round mode=$mode m=$m n=$n"
            numactl -N 1 --preferred=17 "./$binary" "$m" "$n" 13
        done
    done
done
for mode in off final; do
    export TRSM_FUSED_DIAG=0
    [[ $mode == final ]] && export TRSM_FUSED_DIAG=1
    echo "FUSED_FINAL_PROFILE mode=$mode m=17024 n=512"
    TRSM_PROFILE=2 numactl -N 1 --preferred=17 ./fused-final 17024 512 3
done
echo FUSED_CROSSNODE_DONE
