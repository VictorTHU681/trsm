set -euo pipefail
flags='-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp'
gcc $flags bench_trsm.c trsm-owner-cursor-final48-pmu.c -o owner-cursor-final48-pmu -lm -lkblas
for shape in '512 19968' '2432 17024' '17024 512';do
 for flag in 0 1;do
  echo OWNER_CURSOR48_PMU flag=$flag shape=$shape
  TRSM_OWNER_CURSOR=$flag TRSM_PROFILE=2 numactl -N 1 ./owner-cursor-final48-pmu $shape 5
 done
done
for round in 1 2 3 4 5 6;do
 order='0 1';if [ "$((round % 2))" = 0 ];then order='1 0';fi
 for flag in $order;do
  echo OWNER_CURSOR48_SAME round=$round flag=$flag
  TRSM_OWNER_CURSOR=$flag numactl -N 1 ./owner-cursor-final48 512 19968 11
 done
done
for round in 1 2 3 4;do
 order='v48-synced owner-cursor-final48';if [ "$((round % 2))" = 0 ];then order='owner-cursor-final48 v48-synced';fi
 for tag in $order;do
  echo OWNER_CURSOR48_TC3 round=$round tag=$tag
  numactl -N 1 ./$tag 17024 512 11
 done
done
