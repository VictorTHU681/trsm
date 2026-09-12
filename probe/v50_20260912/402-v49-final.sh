set -euo pipefail
flags='-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp'
gcc $flags -c trsm-owner-cursor-final48.c -o cursor48-exact.o
gcc $flags -c trsm-v49-final.c -o v49-final.o
objcopy -O binary --only-section=.text cursor48-exact.o cursor48-exact.text
objcopy -O binary --only-section=.text v49-final.o v49-final.text
cmp cursor48-exact.text v49-final.text
sha256sum v49-final.text trsm-v49-final.c
gcc $flags -DTRSM_A4NT=0 v49-final-boundary.c -o v49-final-boundary-alt -lm
OMP_NUM_THREADS=1 numactl -N 1 --preferred=17 taskset -c 40 ./v49-final-boundary-alt
echo V49_FINAL_TEXT_AND_A4NT_PASS
