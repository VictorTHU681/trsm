#!/usr/bin/env bash
# TRSM 赛题：编译 + 三组测试用例（判题系统直接执行本脚本）
set -o pipefail
cd "$(dirname "$0")" || exit 1

# 不用 set -u：/etc/profile 与 module 脚本会引用未定义变量
source /etc/profile >/dev/null 2>&1 || true
# 编译环境：题面给出的 HPCKit module 路径；若不存在，尝试本集群 HPCKit 的 setvars.sh（含 KML）
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

# 首选：SME(FP64 FMOPA) 内核，需要支持 armv9-a+sme+sme-f64f64 的汇编器；运行期用 HWCAP2 检测硬件，缺失时自动回退。
rm -f trsm_test
if ! gcc -O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp bench_trsm.c trsm.c -o trsm_test -lm -lkblas 2>/dev/null; then
    echo "[run.sh] 编译器不支持 SME 扩展，回退通用实现" >&2
    gcc -O3 -fopenmp -DTRSM_NO_SME_ASM bench_trsm.c trsm.c -o trsm_test -lm -lkblas
fi

# 判题口径：38 线程、单一 NUMA（CPU 节点 1）。绑核在 trsm.c 内部完成；
# 这里的 OMP 变量只让基准程序自带的参考实现（KML dgemm）也能用满核。
export OMP_NUM_THREADS=38
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_DYNAMIC=FALSE
export OMP_MAX_ACTIVE_LEVELS=1
export OMP_WAIT_POLICY=active
export GOMP_SPINCOUNT=infinite
unset GOMP_CPU_AFFINITY OMP_THREAD_LIMIT 2>/dev/null || true

# CPU 节点：默认绑 node 7（判题机为共享机器，避开常见默认 node 1 的共租户争用）；
# 不存在或不可绑时回退 node 1，保证任何拓扑下都能出分。TRSM_CPUNODE 可强制指定。
# 内存：优先使用与所选 CPU 节点相邻的纯内存节点（HBM），放不下时自动回落本地 DDR。
cpunode="${TRSM_CPUNODE:-7}"
numactl -N $cpunode true >/dev/null 2>&1 || cpunode=1
hbm=""
if [[ -r /sys/devices/system/node/node$cpunode/distance ]]; then
    read -ra dist < /sys/devices/system/node/node$cpunode/distance
    best=999
    for n in "${!dist[@]}"; do
        [[ -d /sys/devices/system/node/node$n ]] || continue
        cpulist=$(cat /sys/devices/system/node/node$n/cpulist 2>/dev/null)
        [[ -z "$cpulist" ]] || continue          # 只考虑没有 CPU 的纯内存节点
        if (( dist[n] < best )); then best=${dist[n]}; hbm=$n; fi
    done
    (( best > 70 )) && hbm=""
fi
MEMOPT=""
if [[ -n "$hbm" ]] && numactl -N $cpunode --preferred=$hbm true >/dev/null 2>&1; then
    MEMOPT="--preferred=$hbm"
    # glibc ≥2.35 时让 malloc 的大块 mmap 自带 MADV_HUGEPAGE（提高 X 的透明大页覆盖率）；
    # 2.34 及以下（本集群 Kylin V10 系统库）静默忽略此 tunable，无副作用
    export GLIBC_TUNABLES=glibc.malloc.hugetlb=1
fi
echo "[run.sh] numactl -N $cpunode $MEMOPT" >&2

numactl -N $cpunode $MEMOPT ./trsm_test 512 19968 1
numactl -N $cpunode $MEMOPT ./trsm_test 2432 17024 1
numactl -N $cpunode $MEMOPT ./trsm_test 17024 512 1
