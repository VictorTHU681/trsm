#!/usr/bin/env bash
# val65：全形状生产默认验证闸（用户要求：所有形状必须正常）+ BLAS=1 墙钟风险实测
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S) $(git log --oneline -1)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
rm -f trsm_v65
gcc $FLAGS bench_trsm.c trsm.c -o trsm_v65 -lm -lkblas 2>&1 | grep -E "rror" | head -3
[[ -x trsm_v65 ]] || { echo "BUILD FAILED"; exit 1; }
echo "=== gen 大轮次"
gcc -O3 -fopenmp -DTRSM_NO_SME_ASM -Dl_trsm=l_trsm_ref -c trsm.c -o probe/refv5.o 2>&1 | grep -E "error" || true
gcc $FLAGS probe/gen_test.c trsm.c probe/refv5.o -o probe/genv5 -lm 2>&1 | grep -E "error" || true
echo "[默认60]";   timeout 1200 numactl -N 1 ./probe/genv5 60 2>&1 | tail -1
echo "[owner+steal30]"; TRSM_OWNER=1 TRSM_STEAL=1 timeout 700 numactl -N 1 ./probe/genv5 30 2>&1 | tail -1
echo "[cfold20]";  TRSM_OWNER=1 TRSM_CFOLD=1 timeout 600 numactl -N 1 ./probe/genv5 20 2>&1 | tail -1
echo "[5thr20]";   OMP_NUM_THREADS=5 timeout 600 numactl -N 1 ./probe/genv5 20 2>&1 | tail -1
echo "[1thr10]";   OMP_NUM_THREADS=1 timeout 600 numactl -N 1 ./probe/genv5 10 2>&1 | tail -1
echo "=== 形状矩阵（生产默认，逐个校验）"
for a in "512 19968" "2432 17024" "17024 512" "64 96" "65 97" "63 95" "128 192" "512 96" "512 192" "511 19968" "512 19969" "513 19967" "2433 17023" "2431 17025" "17023 513" "1024 8192" "256 256" "100 20000" "17024 96" "96 17024"; do
  printf "[%s] " "$a"
  OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./trsm_v65 $a 1 2>&1 | awk '/PASS|FAIL/{print $NF; ok=1} END{if(!ok) print "NO-OUTPUT"}' | tail -1
done
echo "=== 线程矩阵（官方三形状 x {1,5,37}）"
for t in 1 5 37; do
  for a in "512 19968" "2432 17024" "17024 512"; do
    printf "[t=%s %s] " "$t" "$a"
    OMP_NUM_THREADS=$t numactl -N 1 $MEMOPT ./trsm_v65 $a 1 2>&1 | awk '/PASS|FAIL/{print $NF; ok=1} END{if(!ok) print "NO-OUTPUT"}' | tail -1
  done
done
echo "=== BLAS_NUM_THREADS=1 墙钟与 wake"
for r in 1 2; do
  echo "--- blas1_tc3_$r  wall_start=$(date +%s.%N)"
  BLAS_NUM_THREADS=1 TRSM_PROFILE=2 OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./trsm_v65 17024 512 4 2>&1 | grep -E " x |PASS|FAIL|唤醒"
  echo "wall_end=$(date +%s.%N)"
  echo "--- noblas_tc3_$r wall_start=$(date +%s.%N)"
  TRSM_PROFILE=2 OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./trsm_v65 17024 512 4 2>&1 | grep -E " x |PASS|FAIL|唤醒"
  echo "wall_end=$(date +%s.%N)"
done
echo "--- blas1 TC1/TC2 冒烟"
BLAS_NUM_THREADS=1 OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./trsm_v65 512 19968 2 2>&1 | grep -E " x |PASS|FAIL"
BLAS_NUM_THREADS=1 OMP_NUM_THREADS=38 numactl -N 1 $MEMOPT ./trsm_v65 2432 17024 2 2>&1 | grep -E " x |PASS|FAIL"
rm -f trsm_v65 probe/refv5.o probe/genv5
echo "=== DONE $(date +%H:%M:%S)"
