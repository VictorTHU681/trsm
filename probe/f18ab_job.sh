#!/usr/bin/env bash
# fresh18 全栈正式认证：master vs fresh18 双二进制交错（判题口径 $MEMOPT），附 TRSM_DNT=1 臂（TC2）与 gen_test 收尾
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S) $(git log --oneline -1)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
git show master:trsm.c > /tmp/trsm_m.c
rm -f trsm_m trsm_f18
gcc $FLAGS bench_trsm.c /tmp/trsm_m.c -o trsm_m -lm -lkblas 2>&1 | grep -E "rror" | head -3
gcc $FLAGS bench_trsm.c trsm.c -o trsm_f18 -lm -lkblas 2>&1 | grep -E "rror" | head -3
[[ -x trsm_m && -x trsm_f18 ]] || { echo COMPILE_FAIL; exit 1; }
run() { local tag="$1" bin="$2"; shift 2; local envs=(); while [[ "$1" != "--" ]]; do envs+=("$1"); shift; done; shift
    env "${envs[@]}" OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./$bin "$1" "$2" "$3" 2>&1 | grep -E "PASS|FAIL" | sed "s/^/[$tag] /"; }
for i in 1 2 3 4 5 6; do
    echo "--- round $i $(date +%H:%M:%S)"
    run "tc2 m  $i" trsm_m   NOOP=0 -- 2432 17024 5
    run "tc2 f  $i" trsm_f18 NOOP=0 -- 2432 17024 5
    run "tc2 fd $i" trsm_f18 TRSM_DNT=1 -- 2432 17024 5
    run "tc1 m  $i" trsm_m   NOOP=0 -- 512 19968 5
    run "tc1 f  $i" trsm_f18 NOOP=0 -- 512 19968 5
    run "tc3 m  $i" trsm_m   NOOP=0 -- 17024 512 3
    run "tc3 f  $i" trsm_f18 NOOP=0 -- 17024 512 3
done
echo "--- gen_test（fresh18）"
gcc -O3 -fopenmp -DTRSM_NO_SME_ASM -Dl_trsm=l_trsm_ref -c trsm.c -o probe/ref.o 2>&1 | grep -E "error"
gcc $FLAGS probe/gen_test.c trsm.c probe/ref.o -o probe/gen_test -lm 2>&1 | grep -E "error"
timeout 900 numactl -N 1 ./probe/gen_test 40 2>&1 | grep -E "FAIL|PASS" | head -3
TRSM_OWNER=1 timeout 600 numactl -N 1 ./probe/gen_test 20 2>&1 | grep -E "FAIL|PASS" | head -3
echo "=== DONE $(date +%H:%M:%S)"
