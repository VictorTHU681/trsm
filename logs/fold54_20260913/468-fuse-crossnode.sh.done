#!/usr/bin/env bash
set -eo pipefail
cd /home/share/niujiayin/trsm-diag55-20260913/generated
export GLIBC_TUNABLES=glibc.malloc.hugetlb=1
sha256sum trsm-diag-fuse.c diag-fuse control
for round in 1 2 3 4 5 6 7 8 9 10 11 12; do
    order=(control off fused)
    [[ $((round%2)) == 0 ]] && order=(fused off control)
    for mode in "${order[@]}"; do
        binary=diag-fuse
        export TRSM_DIAG_SME=0 TRSM_LATE_DIAG=0
        [[ $mode == control ]] && binary=control
        [[ $mode == fused ]] && export TRSM_DIAG_SME=1 TRSM_LATE_DIAG=2
        echo "DIAG_FUSE_FORMAL round=$round mode=$mode m=17024 n=512"
        numactl -N 1 --preferred=17 "./$binary" 17024 512 13
    done
done
echo FUSE_CROSSNODE_DONE
