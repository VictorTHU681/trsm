set -euo pipefail
flags='-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp'
gcc $flags owner-hoist48-boundary.c -o owner-hoist48-boundary -lm
OMP_NUM_THREADS=1 numactl -N 1 --preferred=17 taskset -c 40 ./owner-hoist48-boundary
gcc $flags bench_trsm.c trsm-owner-hoist48.c -o owner-hoist48 -lm -lkblas
for round in 1 2 3 4;do
 order='off on';if [ "$((round % 2))" = 0 ];then order='on off';fi
 for tag in $order;do
  flag=0;if [ "$tag" = on ];then flag=1;fi
  echo OWNER_HOIST48_SCORE round=$round tag=$tag
  TRSM_OWNER_HOIST=$flag numactl -N 1 ./owner-hoist48 512 19968 11
 done
done
for flag in 0 1;do
 echo OWNER_HOIST48_PROFILE flag=$flag
 TRSM_OWNER_HOIST=$flag TRSM_PROFILE=2 numactl -N 1 ./owner-hoist48 512 19968 5
done
