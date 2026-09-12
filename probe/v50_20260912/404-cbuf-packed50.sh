set -euo pipefail
gcc -O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp cbuf-packed50.c -o cbuf-packed50 -lm
OMP_NUM_THREADS=1 numactl -N 1 --preferred=17 taskset -c 40 ./cbuf-packed50
