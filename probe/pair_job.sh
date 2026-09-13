#!/usr/bin/env bash
# 配对任务（TRSM_PAIRP）正确性作业：定向形状 × 环境组合 + 原 gen_test 回归
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== HOST $(hostname) $(date +%H:%M:%S)"
. probe/env.sh
FLAGS="-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp"
gcc -O3 -fopenmp -DTRSM_NO_SME_ASM -Dl_trsm=l_trsm_ref -c trsm.c -o probe/ref.o 2>&1 | grep -E "error" | head -5
gcc $FLAGS probe/pair_gen.c trsm.c probe/ref.o -o probe/pair_gen -lm 2>&1 | grep -E "error" | head -5
gcc $FLAGS probe/gen_test.c trsm.c probe/ref.o -o probe/gen_test -lm 2>&1 | grep -E "error" | head -5
[[ -x probe/pair_gen && -x probe/gen_test ]] || { echo COMPILE_FAIL; exit 1; }

run() { echo "--- [$*]"; env "$@" OMP_NUM_THREADS=38 timeout 900 numactl -N 1 ./probe/pair_gen 2>&1 | grep -vE "^m=.*ok$"; }
# 1. 强制流水打包（JIT_MB=0 → 全部 bigL/ppmode=1）：m>1024 自然进 special=4；npan 偶 → 配对
run TRSM_JIT_MB=0 TRSM_PAIRP=1
# 2. 同上 + 小 m 也走 special（DIRECT=0）
run TRSM_JIT_MB=0 TRSM_PAIRP=1 TRSM_DIRECT=0
# 3. 配对关：同形状对照（验证 PAIRP=0 路径不受任务表改动影响）
run TRSM_JIT_MB=0 TRSM_PAIRP=0
# 4. K 分裂加压（更小分裂阈值 → 大量分裂配对任务 + 归约）
run TRSM_JIT_MB=0 TRSM_PAIRP=1 TRSM_SPLITF=4 TRSM_SPLIT_MINCH=2
run TRSM_JIT_MB=0 TRSM_PAIRP=1 TRSM_SPLITF=2 TRSM_SPLIT_MINCH=2 TRSM_MAXSEG=32
# 5. 少线程（配对任务 + 依赖链在小并行度下）
run TRSM_JIT_MB=0 TRSM_PAIRP=1 OMP_NUM_THREADS=5
# 6. 非 hoist 内核对（duplex_bank4_keep 路径）
run TRSM_JIT_MB=0 TRSM_PAIRP=1 TRSM_ABHOIST=0
# 7. xptm 变体（Xp 列块主序写）
run TRSM_JIT_MB=0 TRSM_PAIRP=1 TRSM_XPTM=1
echo "=== 原 gen_test 回归（默认环境 17 固定 + 40 随机）"
timeout 900 numactl -N 1 ./probe/gen_test 40 2>&1 | tail -1
echo "=== gen_test 强制流水打包"
TRSM_JIT_MB=0 timeout 900 numactl -N 1 ./probe/gen_test 40 2>&1 | tail -1
echo "=== DONE $(date +%H:%M:%S)"
