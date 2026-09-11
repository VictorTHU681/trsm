#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S) $(git log --oneline -1)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
rm -f trsm_test; gcc $FLAGS bench_trsm.c trsm.c -o trsm_test -lm -lkblas 2>&1 | grep -E "rror" | head -5
for args in "17024 512 1" "512 19968 1" "2432 17024 1"; do echo "[$args]"; TRSM_PROBE_REGION=1 TRSM_PROFILE=2 OMP_NUM_THREADS=38 numactl -N 1 ./trsm_test $args 2>&1 | grep -E "唤醒|空并行区|PASS" | tail -3 | cut -c1-120; done
echo "--- strace clone/futex counts (TC3)"
strace -f -c -e trace=clone,clone3,futex,mbind,migrate_pages,move_pages -o /tmp/st.txt env OMP_NUM_THREADS=38 numactl -N 1 ./trsm_test 17024 512 1 2>&1 | grep -E "PASS|FAIL"; cat /tmp/st.txt | head -12
echo "=== DONE $(date +%H:%M:%S)"
