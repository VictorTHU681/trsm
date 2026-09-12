set -euo pipefail
flags='-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp'
gcc $flags bench_trsm.c trsm-v48-synced.c -o v48-synced -lm -lkblas
gcc $flags independent-owner-switch.c trsm-v48-synced.c -o independent-v48-synced -lm
OMP_NUM_THREADS=5 numactl -N 1 ./independent-v48-synced
for round in 1 2 3 4;do
 order='off on';if [ "$((round % 2))" = 0 ];then order='on off';fi
 for tag in $order;do
  flag=0;if [ "$tag" = on ];then flag=1;fi
  echo V48_SYNC_SCORE round=$round tag=$tag shape=512x19968
  TRSM_NEXTA=$flag numactl -N 1 ./v48-synced 512 19968 11
 done
done
for shape in '2432 17024' '17024 512';do
 echo V48_SYNC_SCORE round=1 tag=on shape=$shape
 numactl -N 1 ./v48-synced $shape 11
done
