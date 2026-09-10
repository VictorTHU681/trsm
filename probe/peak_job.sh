#!/usr/bin/env bash
# 理论上限探针作业：计算峰值、HBM/DDR 带宽、KML dtrsm 基线（判题口径 numactl -N 1）
set -uo pipefail
cd "$(dirname "$0")"
echo "=== HOST"; hostname; date
awk '$1=="Cpus_allowed_list:"||$1=="Mems_allowed_list:"' /proc/self/status
set +u; . /work_ssd/software/HPCKit/26.1.RC1/setvars.sh --use-gcc >/dev/null 2>&1; set -u
echo "gcc: $(gcc --version | head -1)"
node=1; hbm=17
echo "=== node$node distance: $(cat /sys/devices/system/node/node$node/distance)"
echo "=== HBM node$hbm:"; grep -E "MemTotal|MemFree" /sys/devices/system/node/node$hbm/meminfo
echo "=== cpufreq: $(cat /sys/devices/system/cpu/cpu38/cpufreq/scaling_cur_freq 2>/dev/null) kHz"

gcc -O2 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp peak.c -o peak || exit 1
gcc -O3 ../bench_trsm.c trsm_kblas.c -o trsm_kblas -lm -lkblas -fopenmp || exit 1
echo "kblas resolved: $(ldd ./trsm_kblas | grep kblas)"

export OMP_NUM_THREADS=38 OMP_PROC_BIND=close OMP_PLACES=cores
echo; echo "########## 峰值 + DDR 带宽（numactl -N $node）"
numactl -N $node ./peak 32
echo; echo "########## HBM 带宽（numactl -N $node -m $hbm）"
numactl -N $node -m $hbm ./peak 32

echo; echo "########## KML dtrsm 基线：判题口径 numactl -N $node（DDR）"
unset OMP_PROC_BIND OMP_PLACES
for args in "512 19968 3" "2432 17024 2" "17024 512 2"; do
    echo "--- $args"; OMP_NUM_THREADS=38 numactl -N $node ./trsm_kblas $args 2>&1 | grep -E "PASS|FAIL|Error" | head -3
done
echo; echo "########## KML dtrsm 基线：--preferred=$hbm（HBM）"
for args in "512 19968 3" "2432 17024 2" "17024 512 2"; do
    echo "--- $args"; OMP_NUM_THREADS=38 numactl -N $node --preferred=$hbm ./trsm_kblas $args 2>&1 | grep -E "PASS|FAIL|Error" | head -3
done
echo; echo "########## KML dtrsm + PROC_BIND（看绑核是否影响库）"
for args in "512 19968 3" "2432 17024 2" "17024 512 2"; do
    echo "--- $args"; OMP_NUM_THREADS=38 OMP_PROC_BIND=close OMP_PLACES=cores numactl -N $node --preferred=$hbm ./trsm_kblas $args 2>&1 | grep -E "PASS|FAIL|Error" | head -3
done
echo "=== DONE"; date
