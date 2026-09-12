set -euo pipefail
flags='-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp'
gcc $flags bench_trsm.c trsm-owner-cursor-final48.c -o owner-cursor-final48 -lm -lkblas
gcc $flags owner-cursor-final48-boundary.c -o owner-cursor-final48-boundary -lm
OMP_NUM_THREADS=1 numactl -N 1 --preferred=17 taskset -c 40 ./owner-cursor-final48-boundary
gcc $flags -c trsm-owner-cursor-final48-audit.c -o owner-cursor-final48-audit.o
gcc $flags gen-strict.c owner-cursor-final48-audit.o ref-removal.o -o strict-owner-cursor48 -lm
numactl -N 1 ./strict-owner-cursor48 20
OMP_NUM_THREADS=5 TRSM_OWNER=1 numactl -N 1 ./strict-owner-cursor48 10
gcc $flags independent-cursor-switch.c owner-cursor-final48-audit.o -o independent-cursor48 -lm
OMP_NUM_THREADS=5 numactl -N 1 ./independent-cursor48
gcc $flags independent-duplex43.c owner-cursor-final48-audit.o -o independent-duplex-cursor48 -lm
OMP_NUM_THREADS=5 TRSM_OWNER=0 TRSM_ANT=1 TRSM_DIRECT=0 TRSM_MB=256 TRSM_SWHOST=0 TRSM_EXPECT_SPECIAL=2 numactl -N 1 ./independent-duplex-cursor48
OMP_NUM_THREADS=5 TRSM_OWNER=0 TRSM_ANT=1 TRSM_DIRECT=0 TRSM_MB=192 TRSM_JIT_MB=0 TRSM_PPMODE=1 TRSM_EXPECT_SPECIAL=4 numactl -N 1 ./independent-duplex-cursor48
gcc $flags validate.c owner-cursor-final48-audit.o -o validate-owner-cursor48 -lm -lkblas
numactl -N 1 ./validate-owner-cursor48
echo OWNER_CURSOR48_VALIDATION_DONE
for round in 1 2 3 4 5 6;do
 order='v48-synced owner-cursor-final48';if [ "$((round % 2))" = 0 ];then order='owner-cursor-final48 v48-synced';fi
 for shape in '512 19968' '2432 17024' '17024 512';do
  for tag in $order;do
   echo OWNER_CURSOR48_SCORE round=$round tag=$tag shape=$shape
   numactl -N 1 ./$tag $shape 11
  done
 done
done
