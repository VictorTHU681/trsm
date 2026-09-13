#!/usr/bin/env bash
set -eo pipefail
cd generated
export GLIBC_TUNABLES=glibc.malloc.hugetlb=1
for shape in '1216 17024 1216' '8512 512 8512'; do
    read -r m n k <<<"$shape"
    for policy in default nt0 interleave mb96 mb192 default; do
        unset FUSED_PACK_NT FUSED_INTERLEAVE FUSED_MB
        case "$policy" in
            nt0) export FUSED_PACK_NT=0;;
            interleave) export FUSED_INTERLEAVE=1;;
            mb96) export FUSED_MB=96;;
            mb192) export FUSED_MB=192;;
        esac
        echo "BATCH_POLICY policy=$policy m=$m n=$n k=$k"
        numactl -N 1 --preferred=17 ./batch-driver "$m" "$n" "$k" 2 1 -1 1 64
    done
done
