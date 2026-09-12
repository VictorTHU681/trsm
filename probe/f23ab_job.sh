#!/usr/bin/env bash
# fresh23（EXTLD 装载 EXT 拼接）正式认证：master vs fresh20 双二进制交错（判题口径），TC3 为主
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S) $(git log --oneline -1)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
git show master:trsm.c > /tmp/trsm_m23.c
rm -f trsm_m trsm_f23
gcc $FLAGS bench_trsm.c /tmp/trsm_m23.c -o trsm_m -lm -lkblas 2>&1 | grep -E "rror" | head -3
gcc $FLAGS bench_trsm.c trsm.c -o trsm_f23 -lm -lkblas 2>&1 | grep -E "rror" | head -3
[[ -x trsm_m && -x trsm_f23 ]] || { echo COMPILE_FAIL; exit 1; }
run() { local tag="$1" bin="$2"; shift 2
    OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./$bin "$1" "$2" "$3" 2>&1 | grep -E "PASS|FAIL" | sed "s/^/[$tag] /"; }
for i in 1 2 3 4 5 6; do
    echo "--- round $i $(date +%H:%M:%S)"
    run "tc1x m $i" trsm_m   512 19968 5
    run "tc1x f $i" trsm_f23 512 19968 5
    run "tc2 m $i" trsm_m   2432 17024 3
    run "tc2 f $i" trsm_f23 2432 17024 3
    run "tc3 m $i" trsm_m   17024 512 3
    run "tc3 f $i" trsm_f23 17024 512 3
done
echo "--- gen_test（fresh20）"
gcc -O3 -fopenmp -DTRSM_NO_SME_ASM -Dl_trsm=l_trsm_ref -c trsm.c -o probe/ref.o 2>&1 | grep -E "error"
gcc $FLAGS probe/gen_test.c trsm.c probe/ref.o -o probe/gen_test -lm 2>&1 | grep -E "error"
timeout 900 numactl -N 1 ./probe/gen_test 60 2>&1 | grep -E "FAIL|PASS" | head -3
TRSM_OWNER=1 timeout 600 numactl -N 1 ./probe/gen_test 20 2>&1 | grep -E "FAIL|PASS" | head -3
echo "=== DONE $(date +%H:%M:%S)"
