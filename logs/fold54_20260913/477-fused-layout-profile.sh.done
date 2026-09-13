#!/usr/bin/env bash
set -eo pipefail
cd generated
export GLIBC_TUNABLES=glibc.malloc.hugetlb=1
for binary in control fused-final; do
    echo "SYMBOL_LAYOUT $binary"
    nm -S --size-sort "$binary" | tail -n 24
done
for round in 1 2; do
    for mode in control final off; do
        binary=fused-final
        unset TRSM_FUSED_DIAG
        [[ $mode == control ]] && binary=control
        [[ $mode == off ]] && export TRSM_FUSED_DIAG=0
        echo "FUSED_FINAL_PROFILE round=$round mode=$mode m=512 n=19968"
        TRSM_PROFILE=2 numactl -N 1 --preferred=17 "./$binary" 512 19968 5
    done
done
echo FUSED_LAYOUT_PROFILE_DONE
