#!/usr/bin/env bash
# v18：HBM 放置审计（判题原样命令）：默认（X/Xp/pp 进 HBM）vs Xp DDR vs pp DDR；run.sh 口径对照；kbench HZ 原型
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST"; hostname; date
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
gcc $FLAGS bench_trsm.c trsm.c -o trsm_test_v18 -lm -lkblas 2>&1 | grep -E "error"; [[ -x trsm_test_v18 ]] || { echo COMPILE_FAIL; exit 1; }
run() {
    local desc="$1"; shift
    echo "--- [$desc]"
    for args in "512 19968 1" "2432 17024 1" "17024 512 1"; do
        env "$@" OMP_NUM_THREADS=38 TRSM_PROFILE=2 numactl -N 1 ./trsm_test_v18 $args 2>&1 | grep -E "PASS|FAIL|prof\] m=|prof\]  每|rror|fault|illed" | awk '/prof\] m=/{h++; if(h%2==0) print; next} /prof\]  每/{p++; if(p%2==0) print; next} {print}' | sed -E "s/每线程平均 ms：prep [0-9.]+ \| //; s/xcopy 0.000 //; s/ \| task 合计.*理想 [0-9.]+ ms，/ | /; s/pack 0.000 //; s/ +/ /g; s/wait 0.00[0-9] //; s/.*tasks=[0-9]+\(items [0-9]+\) nth=38 //; s/ \| setup.*//"
    done
}
run "v18 默认（X/Xp/pp → HBM）"
run "Xp → DDR" TRSM_XPNODE=-2
run "pp → DDR" TRSM_LNODE0=1
run "Xp、pp → DDR（= 旧默认）" TRSM_XPNODE=-2 TRSM_LNODE0=1
run "X 不迁（旧行为）" TRSM_XNODE=-1
run "v18 默认 第二轮"
echo "--- run.sh 口径（--preferred）"
for args in "512 19968 1" "2432 17024 1" "17024 512 1"; do
    OMP_NUM_THREADS=38 TRSM_PROFILE=1 numactl -N 1 $MEMOPT ./trsm_test_v18 $args 2>&1 | grep -E "PASS|FAIL|prof\] m=" | awk '/prof\] m=/{h++; if(h%2==0) print; next} {print}' | sed -E "s/.*tasks=[0-9]+\(items [0-9]+\) nth=38 //; s/ \| setup.*//; s/ +/ /g"
done
echo "=== kbench HZ 原型"
gcc $FLAGS probe/kbench.c -o probe/kbench -lm 2>&1 | grep -E "error"
OMP_NUM_THREADS=1 taskset -c 40 ./probe/kbench 64 128 192 2>&1 | grep -E "^12|^11d|^2h|SIG" | head -6
echo "=== 泛化正确性"
gcc -O3 -fopenmp -DTRSM_NO_SME_ASM -Dl_trsm=l_trsm_ref -c trsm.c -o probe/ref.o 2>&1 | grep -E "error"
gcc $FLAGS probe/gen_test.c trsm.c probe/ref.o -o probe/gen_test -lm 2>&1 | grep -E "error"
timeout 900 numactl -N 1 ./probe/gen_test 40 2>&1 | tail -1
echo "=== DONE"; date
