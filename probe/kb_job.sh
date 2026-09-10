#!/usr/bin/env bash
# 单线程内核微基准
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST"; hostname; date
echo "THP: $(cat /sys/kernel/mm/transparent_hugepage/enabled) defrag: $(cat /sys/kernel/mm/transparent_hugepage/defrag)"
set +u; . /work_ssd/software/HPCKit/26.1.RC1/setvars.sh --use-gcc >/dev/null 2>&1; set -u
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
gcc $FLAGS probe/kbench.c -o probe/kbench -lm 2>&1 | grep -E "error"; [[ -x probe/kbench ]] || { echo KB_COMPILE_FAIL; exit 1; }
node=1; hbm=17
echo "=== kbench（单线程，CPU 40，大数组 DDR）"
OMP_NUM_THREADS=1 taskset -c 40 ./probe/kbench 128 128 256
echo "--- kc=256"
OMP_NUM_THREADS=1 taskset -c 40 ./probe/kbench 256 128 256
echo "--- 大数组放 HBM"
OMP_NUM_THREADS=1 numactl -m $hbm taskset -c 40 ./probe/kbench 128 128 256
echo "=== DONE"; date
