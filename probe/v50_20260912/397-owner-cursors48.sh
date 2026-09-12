set -euo pipefail
flags='-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp'
gcc $flags bench_trsm.c trsm-owner-cursors48.c -o owner-cursors48 -lm -lkblas
for round in 1 2 3 4;do
 order='0 2 3';if [ "$((round % 2))" = 0 ];then order='3 2 0';fi
 for flag in $order;do
  echo OWNER_CURSORS48_SCORE round=$round flag=$flag
  TRSM_OWNER_HOIST=$flag numactl -N 1 ./owner-cursors48 512 19968 11
 done
done
for flag in 0 2 3;do
 echo OWNER_CURSORS48_PROFILE flag=$flag
 TRSM_OWNER_HOIST=$flag TRSM_PROFILE=2 numactl -N 1 ./owner-cursors48 512 19968 5
done
