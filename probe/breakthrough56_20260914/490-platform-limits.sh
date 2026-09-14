#!/usr/bin/env bash
set -eo pipefail
cd generated
gcc -O2 -march=armv9-a+sme+sve2 vector_limits.c -o vector-limits
numactl -N 1 ./vector-limits
lscpu
sha256sum vector_limits.c
echo PLATFORM_LIMITS_DONE
