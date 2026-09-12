set -euo pipefail
flags='-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp'
gcc $flags owner-diagcompact48-boundary.c -o owner-diagcompact48-boundary -lm
OMP_NUM_THREADS=1 numactl -N 1 --preferred=17 taskset -c 40 ./owner-diagcompact48-boundary
gcc $flags bench_trsm.c trsm-owner-diagcompact48.c -o owner-diagcompact48 -lm -lkblas
for round in 1 2 3 4;do
 order='0 1 2';if [ "$((round % 2))" = 0 ];then order='2 1 0';fi
 for flag in $order;do
  echo OWNER_DIAGCOMPACT48_SCORE round=$round flag=$flag
  TRSM_ODIAG=$flag numactl -N 1 ./owner-diagcompact48 512 19968 11
 done
done
for flag in 0 1 2;do
 echo OWNER_DIAGCOMPACT48_PROFILE flag=$flag
 TRSM_ODIAG=$flag TRSM_PROFILE=2 numactl -N 1 ./owner-diagcompact48 512 19968 5
done
