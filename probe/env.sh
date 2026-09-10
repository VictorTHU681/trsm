# 供作业脚本 source：编译环境 + 运行时探测 HBM 节点（同 run.sh 的逻辑）
set +u; . /work_ssd/software/HPCKit/26.1.RC1/setvars.sh --use-gcc >/dev/null 2>&1; set -u
node=1
hbm=""
if [[ -r /sys/devices/system/node/node$node/distance ]]; then
    read -ra dist < /sys/devices/system/node/node$node/distance
    best=999
    for n in "${!dist[@]}"; do
        [[ -d /sys/devices/system/node/node$n ]] || continue
        cpulist=$(cat /sys/devices/system/node/node$n/cpulist 2>/dev/null)
        [[ -z "$cpulist" ]] || continue
        if (( dist[n] < best )); then best=${dist[n]}; hbm=$n; fi
    done
    (( best > 70 )) && hbm=""
fi
MEMOPT=""
if [[ -n "$hbm" ]] && numactl -N $node --preferred=$hbm true >/dev/null 2>&1; then MEMOPT="--preferred=$hbm"; fi
echo "[env] host=$(hostname) hbm=${hbm:-none} MEMOPT=$MEMOPT nodes=$(numactl -H | head -1)"
export OMP_NUM_THREADS=38 OMP_PROC_BIND=close OMP_PLACES=cores OMP_DYNAMIC=FALSE OMP_WAIT_POLICY=active GOMP_SPINCOUNT=infinite
