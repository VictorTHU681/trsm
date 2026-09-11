#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S) $(git log --oneline -1)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
rm -f trsm_test; gcc $FLAGS bench_trsm.c trsm.c -o trsm_test -lm -lkblas 2>&1 | grep -E "rror" | head -5
ldd ./trsm_test | grep -i "blas\|omp\|pthread"
strings $(ldd ./trsm_test | awk '/kblas/{print $3}') 2>/dev/null | grep -E "^(OPENBLAS|KBLAS|GOTO)_[A-Z_]*THREADS|THREAD_TIMEOUT|OPENBLAS_[A-Z_]+" | sort -u | head -12
for v in "X=0" "OPENBLAS_NUM_THREADS=1" "KBLAS_NUM_THREADS=1" "OMP_WAIT_POLICY=passive"; do echo "[$v]"; env $v TRSM_PROFILE=2 OMP_NUM_THREADS=38 numactl -N 1 ./trsm_test 17024 512 1 2>&1 | grep -E "唤醒|PASS" | tail -2 | cut -c1-120; done
echo "--- thread census during TC3 run"
( TRSM_PROFILE=0 OMP_NUM_THREADS=38 numactl -N 1 ./trsm_test 17024 512 1 >/dev/null 2>&1 & pid=$!; sleep 2.5; ps -L -o tid,psr,pcpu,comm -p $pid 2>/dev/null | awk 'NR>1{n++; if($3>5) busy++} END{print "threads="n" busy="busy}'; ps -L -o tid,psr,pcpu,stat -p $pid 2>/dev/null | sort -k3 -n -r | head -5; wait $pid )
for i in 1 2 3; do for v in "X=0" "OPENBLAS_NUM_THREADS=1"; do printf "%-26s " "[$v]"; env $v OMP_NUM_THREADS=38 numactl -N 1 ./trsm_test 17024 512 1 2>&1 | grep -E "PASS|FAIL"; done; done
echo "=== DONE $(date +%H:%M:%S)"
