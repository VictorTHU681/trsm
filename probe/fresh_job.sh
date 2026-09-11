#!/usr/bin/env bash
# fresh 分支起点：三用例分相 profile + 不带先验的块参数扫描（通用路径，SPECIAL=0 与默认各测）
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
rm -f fastbench; gcc $FLAGS probe/fastbench.c trsm.c -o fastbench -lm 2>&1 | grep -E "rror" | head -5; [[ -x fastbench ]] || { echo COMPILE_FAIL; exit 1; }
run() { local mn="$1"; shift; echo "--- [$mn] [$*]"; env "$@" OMP_NUM_THREADS=38 numactl -N 1 ./fastbench -r 3 $mn 2>&1 | grep -vE "^  rep"; }
prof() { local mn="$1"; shift; echo "--- prof [$mn] [$*]"; env "$@" TRSM_PROFILE=2 OMP_NUM_THREADS=38 numactl -N 1 ./fastbench -r 1 $mn 2>&1 | grep -E "prof\]|min " | tail -3; }
echo "=== 分相 profile（默认配置）"
prof "512 19968" A=1
prof "2432 17024" A=1
prof "17024 512" A=1
prof "2432 17024" TRSM_SPECIAL=0
prof "17024 512" TRSM_SPECIAL=0
echo "=== case2 块参数扫描（通用路径）"
C2="2432 17024"
run "$C2" A=1
run "$C2" TRSM_SPECIAL=0
for cfg in "256 128 64" "256 128 128" "256 192 64" "256 256 64" "320 128 64" "384 128 64" "384 128 128" "192 192 64" "192 256 64" "128 256 64" "128 256 128" "512 128 64" "512 64 64" "256 96 64" "256 64 64"; do
    set -- $cfg; run "$C2" TRSM_MB=$1 TRSM_NC=$2 TRSM_KC=$3
done
echo "=== case3 块参数扫描（通用路径）"
C3="17024 512"
run "$C3" A=1
run "$C3" TRSM_SPECIAL=0
for cfg in "192 128 64" "192 128 192" "256 128 64" "256 128 128" "256 128 256" "384 128 64" "384 128 128" "512 128 64" "512 128 128" "192 256 64" "256 256 64" "128 512 64" "256 512 64" "192 64 64"; do
    set -- $cfg; run "$C3" TRSM_MB=$1 TRSM_NC=$2 TRSM_KC=$3
done
echo "=== case3：ppmode / jit"
run "$C3" TRSM_PPMODE=0
run "$C3" TRSM_JIT=1
run "$C3" TRSM_JIT_MB=2048
echo "=== DONE $(date +%H:%M:%S)"
