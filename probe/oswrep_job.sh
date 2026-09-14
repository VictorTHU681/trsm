#!/usr/bin/env bash
# oswrep：侦探复现配方验证——旧守卫(HEAD~1)应 FAIL、新守卫(HEAD)应 PASS；STEAL=0 差分应 PASS
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S) $(git log --oneline -1)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
git show 95f4b5b:trsm.c > /tmp/trsm_old_$$.c
rm -f trsm_new trsm_oldg
gcc $FLAGS bench_trsm.c trsm.c -o trsm_new -lm -lkblas 2>&1 | grep -E "rror" | head -3
gcc $FLAGS bench_trsm.c /tmp/trsm_old_$$.c -o trsm_oldg -lm -lkblas 2>&1 | grep -E "rror" | head -3
[[ -x trsm_new && -x trsm_oldg ]] || { echo "BUILD FAILED"; exit 1; }
rep(){ local lbl="$1" bin="$2" thr="$3" m="$4" n="$5"; shift 5; printf "[%s] " "$lbl"; env "$@" TRSM_OWNER=1 TRSM_OSW=1 OMP_NUM_THREADS=$thr numactl -N 1 $MEMOPT ./$bin $m $n 1 2>&1 | awk '/PASS|FAIL/{print $NF; ok=1} END{if(!ok)print "NO-OUTPUT"}' | tail -1; }
echo "=== 复现形状（旧守卫预期 FAIL）"
rep "old_128x768_t3"  trsm_oldg 3 128 768
rep "old_192x960_t3"  trsm_oldg 3 192 960
rep "old_512x2496_t8" trsm_oldg 8 512 2496
rep "old_tc1_t38"     trsm_oldg 38 512 19968
echo "=== 同形状 STEAL=0 差分（旧守卫预期 PASS）"
rep "old_128x768_nosteal" trsm_oldg 3 128 768 TRSM_STEAL=0
echo "=== 新守卫（预期全 PASS）"
rep "new_128x768_t3"  trsm_new 3 128 768
rep "new_192x960_t3"  trsm_new 3 192 960
rep "new_512x2496_t8" trsm_new 8 512 2496
rep "new_tc1_t38"     trsm_new 38 512 19968
echo "=== 生产默认冒烟（OSW 关，两码应全 PASS）"
for a in "512 19968" "2432 17024" "17024 512"; do printf "[def %s] " "$a"; OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./trsm_new $a 1 2>&1 | awk '/PASS|FAIL/{print $NF}' | tail -1; done
rm -f trsm_new trsm_oldg /tmp/trsm_old_$$.c
echo "=== DONE $(date +%H:%M:%S)"
