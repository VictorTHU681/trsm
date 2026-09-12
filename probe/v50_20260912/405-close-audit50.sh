set -euo pipefail
flags='-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp'
sha256sum trsm-v49-final.c bench_trsm.c
gcc $flags close-input-audit50.c trsm-close-audit50.c -o close-input-audit50 -lm -lkblas
numactl -N 1 ./close-input-audit50
gcc $flags bench_trsm.c trsm-v49-final.c -o v50-close-production -lm -lkblas
for shape in '512 19968' '2432 17024' '17024 512';do
 echo CLOSE50_FORMAL shape=$shape
 numactl -N 1 ./v50-close-production $shape 1
done
echo CLOSE50_DONE
