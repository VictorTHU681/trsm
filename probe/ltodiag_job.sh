#!/usr/bin/env bash
# 诊断：run.sh 自带环境下 LTO 编译为何失败（不带 2>/dev/null 抓真错）
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S)"
source /etc/profile >/dev/null 2>&1 || true
if [[ -d /home/HPC/HPCKit/latest/modulefiles ]]; then
    module use /home/HPC/HPCKit/latest/modulefiles 2>/dev/null
    module load gcc/compiler12.3.1/gccmodule 2>/dev/null
    module load gcc/kml25.2.0/kblas/multi 2>/dev/null
    echo "path=module"
else
    for p in /work_ssd/software/HPCKit/latest/setvars.sh /work_ssd/software/HPCKit/26.1.RC1/setvars.sh \
             /home/share/eapp_sysu/HPCKit/latest/setvars.sh /home/share/eapp_sysu/HPCKit/26.1.RC1/setvars.sh; do
        if [[ -f "$p" ]]; then . "$p" --use-gcc >/dev/null 2>&1; echo "path=setvars:$p"; break; fi
    done
fi
echo "gcc=$(which gcc) $(gcc --version 2>/dev/null | head -1)"
echo "ld=$(which ld) $(ld --version 2>/dev/null | head -1)"
echo "--- base 编译"
gcc -O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp bench_trsm.c trsm.c -o /tmp/t_b_$$ -lm -lkblas 2>&1 | head -5; echo "rc=$?"
echo "--- LTO 编译（抓错）"
gcc -O3 -march=armv9-a+sme+sme-f64f64+sve2 -mtune=tsv110 -flto -fopenmp bench_trsm.c trsm.c -o /tmp/t_l_$$ -lm -lkblas 2>&1 | head -25; echo "rc=$?"
rm -f /tmp/t_b_$$ /tmp/t_l_$$
echo "=== DONE"
