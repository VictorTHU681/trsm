#!/usr/bin/env bash
set -eo pipefail
# Use the already-built, immutable candidate and control from request 462.
cd /home/share/niujiayin/trsm-diag55-20260913/generated
export GLIBC_TUNABLES=glibc.malloc.hugetlb=1
sha256sum trsm-v54.c trsm-diag-final.c control diag-final
cat /sys/devices/system/cpu/cpu40/cpufreq/scaling_cur_freq
for shape in '512 19968' '2432 17024' '17024 512'; do
    read -r m n <<<"$shape"
    for round in 1 2 3 4 5 6 7 8 9 10 11 12; do
        order=(control final off)
        [[ $((round%2)) == 0 ]] && order=(off final control)
        for mode in "${order[@]}"; do
            binary=diag-final
            unset TRSM_DIAG_SME
            [[ $mode == control ]] && binary=control
            [[ $mode == off ]] && export TRSM_DIAG_SME=0
            echo "DIAG_FINAL_FORMAL round=$round mode=$mode m=$m n=$n"
            numactl -N 1 --preferred=17 "./$binary" "$m" "$n" 13
        done
    done
done
echo DIAG_FINAL_CROSSNODE_DONE
