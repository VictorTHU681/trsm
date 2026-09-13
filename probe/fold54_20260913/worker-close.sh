#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname "$0")"
source /etc/profile >/dev/null 2>&1 || true
kit=/work_ssd/software/HPCKit/26.1.RC1/modulefiles
module use "$kit"
compiler=("$kit"/gcc/compiler*/gccmodule)
blas=("$kit"/gcc/kml*/kblas/multi)
module load "${compiler[0]#"$kit/"}"
module load "${blas[0]#"$kit/"}"
export OMP_NUM_THREADS=38 OMP_PROC_BIND=close OMP_PLACES=cores
export OMP_DYNAMIC=FALSE OMP_MAX_ACTIVE_LEVELS=1
export OMP_WAIT_POLICY=active GOMP_SPINCOUNT=infinite
unset GOMP_CPU_AFFINITY OMP_THREAD_LIMIT
hostname
date -Iseconds
gcc --version
numactl -H
mkdir -p requests results
deadline=$((SECONDS + 900))
while [[ ! -e STOP && $SECONDS -lt $deadline ]]; do
    for task in requests/*.sh; do
        [[ -e "$task" ]] || continue
        name=${task##*/}
        mv "$task" "results/$name.running"
        timeout 1200 bash "results/$name.running" > "results/$name.log" 2>&1 && status=0 || status=$?
        echo "$status" > "results/$name.status"
        mv "results/$name.running" "results/$name.done"
    done
    sleep 1
done
