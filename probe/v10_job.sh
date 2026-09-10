#!/usr/bin/env bash
# v10：kc=64 全程托管回写（kloop_epi64）；kbench 单线程核验 + 三用例 + 泛化正确性 + 变体对比
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST"; hostname; date
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
gcc $FLAGS probe/kbench.c -o probe/kbench -lm 2>&1 | grep -E "error"; [[ -x probe/kbench ]] || { echo KB_COMPILE_FAIL; exit 1; }
gcc $FLAGS -DTRSM_HOSTPF=0 probe/kbench.c -o probe/kbench_nopf -lm 2>&1 | grep -E "error"
echo "=== kbench kc=64 nc=128 mb=192（第 1/2 节）"
OMP_NUM_THREADS=1 taskset -c 40 ./probe/kbench 64 128 192 2>&1 | grep -E "^1\.|^1[a-z]\.|^2|SIG|^kc" | head -20
echo "--- HOSTPF=0"
OMP_NUM_THREADS=1 taskset -c 40 ./probe/kbench_nopf 64 128 192 2>&1 | grep -E "^2h|^2s|SIG" | head -4
gcc $FLAGS bench_trsm.c trsm.c -o trsm_test_v10 -lm -lkblas 2>&1 | grep -E "error"; [[ -x trsm_test_v10 ]] || { echo COMPILE_FAIL; exit 1; }
gcc $FLAGS -DTRSM_HOSTPF=0 bench_trsm.c trsm.c -o trsm_test_v10np -lm -lkblas 2>&1 | grep -E "error"
run3() {
    local bin="$1" desc="$2"; shift 2
    echo "--- [$desc]"
    for args in "512 19968 1" "2432 17024 1" "17024 512 1"; do
        env "$@" TRSM_PROFILE=2 numactl -N $node $MEMOPT ./$bin $args 2>&1 | grep -E "PASS|FAIL|prof\]  |rror|fault|illed" | tail -2
    done
}
run3 trsm_test_v10 "v10 默认 mb192 nc128 kc64（托管步带预取）"
run3 trsm_test_v10np "v10 HOSTPF=0（托管步不带预取）"
run3 trsm_test_v10 "v10 mb256 kc64" TRSM_MB=256
run3 trsm_test_v10 "v10 mb192 kc64 splitf=6" TRSM_SPLITF=6
run3 trsm_test_v10 "v10 mb192 kc64 nosplit" TRSM_SPLITF=0
run3 trsm_test_v10 "v10 mb192 kc96（对照）" TRSM_KC=96
echo "=== 泛化正确性"
gcc -O3 -fopenmp -DTRSM_NO_SME_ASM -Dl_trsm=l_trsm_ref -c trsm.c -o probe/ref.o 2>&1 | grep -E "error"
gcc $FLAGS probe/gen_test.c trsm.c probe/ref.o -o probe/gen_test -lm 2>&1 | grep -E "error"
timeout 900 numactl -N $node ./probe/gen_test 60 2>&1 | tail -1
TRSM_KC=64 TRSM_MB=64 timeout 600 numactl -N $node ./probe/gen_test 30 2>&1 | tail -1
echo "=== DONE"; date
