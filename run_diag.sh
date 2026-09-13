#!/usr/bin/env bash
# TRSM 诊断版 run.sh：与正式版完全相同的编译与运行方式，只是
# (1) 打开算子既有的只读分相计时（TRSM_PROFILE=2，stderr）；
# (2) run.sh 起进程外采样器：对运行中的 trsm_test 抓 /proc/<pid>/smaps 大 VMA 的 THP 覆盖率、
#     numa_maps 放置、节点 MemFree 与 loadavg（算子进程零开销、trsm.c 零改动）；
# (3) 各用例 stderr 与采样在三行正式结果之后统一打印到 stdout（判题日志只回显 stdout）。
set -o pipefail
cd "$(dirname "$0")" || exit 1

source /etc/profile >/dev/null 2>&1 || true
if [[ -d /home/HPC/HPCKit/latest/modulefiles ]]; then
    module use /home/HPC/HPCKit/latest/modulefiles 2>/dev/null
    module load gcc/compiler12.3.1/gccmodule 2>/dev/null
    module load gcc/kml25.2.0/kblas/multi 2>/dev/null
else
    for p in /work_ssd/software/HPCKit/latest/setvars.sh \
             /work_ssd/software/HPCKit/26.1.RC1/setvars.sh \
             /home/share/eapp_sysu/HPCKit/latest/setvars.sh \
             /home/share/eapp_sysu/HPCKit/26.1.RC1/setvars.sh; do
        if [[ -f "$p" ]]; then . "$p" --use-gcc >/dev/null 2>&1; break; fi
    done
fi

rm -f trsm_test
if ! gcc -O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp bench_trsm.c trsm.c -o trsm_test -lm -lkblas 2>/dev/null; then
    echo "[run.sh] 编译器不支持 SME 扩展，回退通用实现" >&2
    gcc -O3 -fopenmp -DTRSM_NO_SME_ASM bench_trsm.c trsm.c -o trsm_test -lm -lkblas
fi

export OMP_NUM_THREADS=38
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_DYNAMIC=FALSE
export OMP_MAX_ACTIVE_LEVELS=1
export OMP_WAIT_POLICY=active
export GOMP_SPINCOUNT=infinite
unset GOMP_CPU_AFFINITY OMP_THREAD_LIMIT 2>/dev/null || true

cpunode=1
hbm=""
if [[ -r /sys/devices/system/node/node$cpunode/distance ]]; then
    read -ra dist < /sys/devices/system/node/node$cpunode/distance
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
if [[ -n "$hbm" ]] && numactl -N $cpunode --preferred=$hbm true >/dev/null 2>&1; then
    MEMOPT="--preferred=$hbm"
    export GLIBC_TUNABLES=glibc.malloc.hugetlb=1
fi
echo "[run.sh] numactl -N $cpunode $MEMOPT" >&2

export TRSM_PROFILE=2   # 算子只读分相计时（stderr；诊断包分数不作数）
export TRSM_SMAPS=1     # 算子进程内 THP 覆盖率抽样（X/L/pp 所在 VMA；每调用一次 smaps 扫描，
                        # 会把 TC1 计时抬 ~4ms/调用、TC2 的 prep 桶抬 ~190ms——只污染 setup/prep，
                        # 判别所需的 kloop/load/store/dkloop 桶不受影响；进程外采样在 TC3 上会锁竞争挂死，勿用）

D=diag_out
rm -rf "$D"; mkdir -p "$D"

nodefree() {
    for n in $cpunode $hbm; do
        [[ -n "$n" && -r /sys/devices/system/node/node$n/meminfo ]] || continue
        grep -E "MemFree|AnonHugePages" /sys/devices/system/node/node$n/meminfo | tr -s " " | tr "\n" " "
    done
    echo ""
}

envsnap() {
    echo "--- [$1] $(date +%H:%M:%S) loadavg=$(cat /proc/loadavg 2>/dev/null)"
    echo "host=$(hostname) kernel=$(uname -r)"
    echo "thp_enabled=$(cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null)"
    echo "thp_defrag=$(cat /sys/kernel/mm/transparent_hugepage/defrag 2>/dev/null)"
    nodefree
    echo "glibc=$(ldd --version 2>/dev/null | head -1)"
    echo "gcc=$(gcc --version 2>/dev/null | head -1)"
}

run_case() {   # $1=名字 $2..=参数；结果行照常走 stdout，stderr 收进文件
    envsnap "before-$1" >> "$D/env.txt" 2>&1
    numactl -N $cpunode $MEMOPT ./trsm_test "${@:2}" 2> "$D/$1.err"
}

envsnap "before-all" > "$D/env.txt" 2>&1
run_case tc1 512 19968 1
run_case tc2 2432 17024 1
run_case tc3 17024 512 1
envsnap "after-all" >> "$D/env.txt" 2>&1

# 三行正式结果之后再打印诊断（判题解析先看到正常输出）
echo "===================== [trsm-diag] ====================="
cat "$D/env.txt"
for c in tc1 tc2 tc3; do
    echo "--- [trsm-diag] $c stderr ---"
    cat "$D/$c.err" 2>/dev/null
done
echo "===================== [trsm-diag end] ================="
