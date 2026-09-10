#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

source /etc/profile >/dev/null 2>&1 || true
if [[ ! -d /home/HPC/HPCKit/latest/modulefiles ]]; then
    echo "错误：缺少 /home/HPC/HPCKit/latest/modulefiles，无法加载 KBLAS。" >&2
    exit 1
fi
module use /home/HPC/HPCKit/latest/modulefiles
module load gcc/compiler12.3.1/gccmodule
module load gcc/kml25.2.0/kblas/multi

gcc -O3 bench_trsm.c trsm.c -o trsm_test -lm -lkblas -fopenmp

export OMP_NUM_THREADS=38
export OMP_DYNAMIC=FALSE
export OMP_MAX_ACTIVE_LEVELS=1
numactl -N 1 ./trsm_test 512 19968 1
numactl -N 1 ./trsm_test 2432 17024 1
numactl -N 1 ./trsm_test 17024 512 1
