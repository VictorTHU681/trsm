set -euo pipefail
flags='-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp'
gcc $flags owner-diagduplex50-boundary.c -o owner-diagduplex50-boundary -lm
OMP_NUM_THREADS=1 numactl -N 1 --preferred=17 taskset -c 40 ./owner-diagduplex50-boundary
gcc $flags bench_trsm.c trsm-owner-diagduplex50.c -o owner-diagduplex50 -lm -lkblas
for round in 1 2 3 4;do
 order='0 1';if [ "$((round % 2))" = 0 ];then order='1 0';fi
 for flag in $order;do
  echo OWNER_DIAGDUP50_SCORE round=$round flag=$flag
  TRSM_ODIAGDUP=$flag numactl -N 1 ./owner-diagduplex50 512 19968 11
 done
done
for flag in 0 1;do
 echo OWNER_DIAGDUP50_PROFILE flag=$flag
 TRSM_ODIAGDUP=$flag TRSM_PROFILE=2 numactl -N 1 ./owner-diagduplex50 512 19968 5
done
