// probe/ddiag.c —— TC2 对角内核：16x16 双 bank + K64 分块 ZA 驻留 vs 现行 compact 单 bank
// （Fable-TC2 蓝图 §3；宏由生产 DUPLEX0/4_U_* 程序化派生：SET=删排水 ld/fsub、st 存 mova 值；PURE=排水槽全删）
// 判据：加权差 ≤−8% GO；b4 桶 ≈ b0 桶（L1 重读成立）；bitwise PASS（fp 逐元素 k 序不变）。
// 编译：gcc -O2 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp probe/ddiag.c -o probe/ddiag -lm
// 运行：numactl -N 1 taskset -c 40 ./probe/ddiag
#include "../trsm.c"

#define DDSET0_S_BODY(BS, CS, C8, C864, ALD) \
 "mova z16.d, p0/m, za4h.d[w12, 0]\n\t" \
 "mova z18.d, p0/m, za5h.d[w12, 0]\n\t" \
 "mova z20.d, p0/m, za6h.d[w12, 0]\n\t" \
 "mova z22.d, p0/m, za7h.d[w12, 0]\n\t" \
 ALD "z8.d, p0/z, [%[a]]\n\t" \
 "fmopa za0.d, p0/m, p0/m, z0.d, z4.d\n\t" \
 ALD "z9.d, p0/z, [%[a], #1, mul vl]\n\t" \
 "fmopa za1.d, p0/m, p0/m, z0.d, z5.d\n\t" \
 "ld1d z12.d, p0/z, [%[b]]\n\t" \
 "fmopa za2.d, p0/m, p0/m, z1.d, z4.d\n\t" \
 "ld1d z13.d, p0/z, [%[b], #1, mul vl]\n\t" \
 "fmopa za3.d, p0/m, p0/m, z1.d, z5.d\n\t" \
 "add %[a], %[a], #128\n\t" \
 "add %[b], %[b], #" #BS "\n\t" \
 ALD "z0.d, p0/z, [%[a]]\n\t" \
 "fmopa za0.d, p0/m, p0/m, z8.d, z12.d\n\t" \
 ALD "z1.d, p0/z, [%[a], #1, mul vl]\n\t" \
 "fmopa za1.d, p0/m, p0/m, z8.d, z13.d\n\t" \
 "ld1d z4.d, p0/z, [%[b]]\n\t" \
 "fmopa za2.d, p0/m, p0/m, z9.d, z12.d\n\t" \
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t" \
 "fmopa za3.d, p0/m, p0/m, z9.d, z13.d\n\t" \
 "add %[a], %[a], #128\n\t" \
 "add %[b], %[b], #" #BS "\n\t" \
 "st1d z16.d, p0, [%[ct], #0, mul vl]\n\t" \
 "prfm pldl2keep, [%[ap]]\n\t" \
 "prfm pldl2keep, [%[bp]]\n\t" \
 "add %[ap], %[ap], #64\n\t" \
 "add %[bp], %[bp], #64\n\t" \
 ALD "z8.d, p0/z, [%[a]]\n\t" \
 "fmopa za0.d, p0/m, p0/m, z0.d, z4.d\n\t" \
 ALD "z9.d, p0/z, [%[a], #1, mul vl]\n\t" \
 "fmopa za1.d, p0/m, p0/m, z0.d, z5.d\n\t" \
 "ld1d z12.d, p0/z, [%[b]]\n\t" \
 "fmopa za2.d, p0/m, p0/m, z1.d, z4.d\n\t" \
 "ld1d z13.d, p0/z, [%[b], #1, mul vl]\n\t" \
 "fmopa za3.d, p0/m, p0/m, z1.d, z5.d\n\t" \
 "add %[a], %[a], #128\n\t" \
 "add %[b], %[b], #" #BS "\n\t" \
 ALD "z0.d, p0/z, [%[a]]\n\t" \
 "fmopa za0.d, p0/m, p0/m, z8.d, z12.d\n\t" \
 ALD "z1.d, p0/z, [%[a], #1, mul vl]\n\t" \
 "fmopa za1.d, p0/m, p0/m, z8.d, z13.d\n\t" \
 "ld1d z4.d, p0/z, [%[b]]\n\t" \
 "fmopa za2.d, p0/m, p0/m, z9.d, z12.d\n\t" \
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t" \
 "fmopa za3.d, p0/m, p0/m, z9.d, z13.d\n\t" \
 "add %[a], %[a], #128\n\t" \
 "add %[b], %[b], #" #BS "\n\t" \
 "st1d z18.d, p0, [%[ct], #1, mul vl]\n\t" \
 "prfm pldl2keep, [%[pc]]\n\t" \
 "prfm pldl2keep, [%[pc], #64]\n\t" \
 "prfm pldl2keep, [%[pc], #" #C8 "]\n\t" \
 "prfm pldl2keep, [%[pc], #" #C864 "]\n\t" \
 "add %[pc], %[pc], #" #CS "\n\t" \
 ALD "z8.d, p0/z, [%[a]]\n\t" \
 "fmopa za0.d, p0/m, p0/m, z0.d, z4.d\n\t" \
 ALD "z9.d, p0/z, [%[a], #1, mul vl]\n\t" \
 "fmopa za1.d, p0/m, p0/m, z0.d, z5.d\n\t" \
 "ld1d z12.d, p0/z, [%[b]]\n\t" \
 "fmopa za2.d, p0/m, p0/m, z1.d, z4.d\n\t" \
 "ld1d z13.d, p0/z, [%[b], #1, mul vl]\n\t" \
 "fmopa za3.d, p0/m, p0/m, z1.d, z5.d\n\t" \
 "add %[a], %[a], #128\n\t" \
 "add %[b], %[b], #" #BS "\n\t" \
 ALD "z0.d, p0/z, [%[a]]\n\t" \
 "fmopa za0.d, p0/m, p0/m, z8.d, z12.d\n\t" \
 ALD "z1.d, p0/z, [%[a], #1, mul vl]\n\t" \
 "fmopa za1.d, p0/m, p0/m, z8.d, z13.d\n\t" \
 "ld1d z4.d, p0/z, [%[b]]\n\t" \
 "fmopa za2.d, p0/m, p0/m, z9.d, z12.d\n\t" \
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t" \
 "fmopa za3.d, p0/m, p0/m, z9.d, z13.d\n\t" \
 "add %[a], %[a], #128\n\t" \
 "add %[b], %[b], #" #BS "\n\t" \
 "st1d z20.d, p0, [%[cb], #0, mul vl]\n\t" \
 "prfm pldl2keep, [%[ap]]\n\t" \
 "prfm pldl2keep, [%[bp]]\n\t" \
 "add %[ap], %[ap], #64\n\t" \
 "add %[bp], %[bp], #64\n\t" \
 ALD "z8.d, p0/z, [%[a]]\n\t" \
 "fmopa za0.d, p0/m, p0/m, z0.d, z4.d\n\t" \
 ALD "z9.d, p0/z, [%[a], #1, mul vl]\n\t" \
 "fmopa za1.d, p0/m, p0/m, z0.d, z5.d\n\t" \
 "ld1d z12.d, p0/z, [%[b]]\n\t" \
 "fmopa za2.d, p0/m, p0/m, z1.d, z4.d\n\t" \
 "ld1d z13.d, p0/z, [%[b], #1, mul vl]\n\t" \
 "fmopa za3.d, p0/m, p0/m, z1.d, z5.d\n\t" \
 "add %[a], %[a], #128\n\t" \
 "add %[b], %[b], #" #BS "\n\t"

#define DDPURE0_P_BODY(BS, CS, C8, C864, ALD) \
 ALD "z8.d, p0/z, [%[a]]\n\t" \
 "fmopa za0.d, p0/m, p0/m, z0.d, z4.d\n\t" \
 ALD "z9.d, p0/z, [%[a], #1, mul vl]\n\t" \
 "fmopa za1.d, p0/m, p0/m, z0.d, z5.d\n\t" \
 "ld1d z12.d, p0/z, [%[b]]\n\t" \
 "fmopa za2.d, p0/m, p0/m, z1.d, z4.d\n\t" \
 "ld1d z13.d, p0/z, [%[b], #1, mul vl]\n\t" \
 "fmopa za3.d, p0/m, p0/m, z1.d, z5.d\n\t" \
 "add %[a], %[a], #128\n\t" \
 "add %[b], %[b], #" #BS "\n\t" \
 ALD "z0.d, p0/z, [%[a]]\n\t" \
 "fmopa za0.d, p0/m, p0/m, z8.d, z12.d\n\t" \
 ALD "z1.d, p0/z, [%[a], #1, mul vl]\n\t" \
 "fmopa za1.d, p0/m, p0/m, z8.d, z13.d\n\t" \
 "ld1d z4.d, p0/z, [%[b]]\n\t" \
 "fmopa za2.d, p0/m, p0/m, z9.d, z12.d\n\t" \
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t" \
 "fmopa za3.d, p0/m, p0/m, z9.d, z13.d\n\t" \
 "add %[a], %[a], #128\n\t" \
 "add %[b], %[b], #" #BS "\n\t" \
 "prfm pldl2keep, [%[ap]]\n\t" \
 "prfm pldl2keep, [%[bp]]\n\t" \
 "add %[ap], %[ap], #64\n\t" \
 "add %[bp], %[bp], #64\n\t" \
 ALD "z8.d, p0/z, [%[a]]\n\t" \
 "fmopa za0.d, p0/m, p0/m, z0.d, z4.d\n\t" \
 ALD "z9.d, p0/z, [%[a], #1, mul vl]\n\t" \
 "fmopa za1.d, p0/m, p0/m, z0.d, z5.d\n\t" \
 "ld1d z12.d, p0/z, [%[b]]\n\t" \
 "fmopa za2.d, p0/m, p0/m, z1.d, z4.d\n\t" \
 "ld1d z13.d, p0/z, [%[b], #1, mul vl]\n\t" \
 "fmopa za3.d, p0/m, p0/m, z1.d, z5.d\n\t" \
 "add %[a], %[a], #128\n\t" \
 "add %[b], %[b], #" #BS "\n\t" \
 ALD "z0.d, p0/z, [%[a]]\n\t" \
 "fmopa za0.d, p0/m, p0/m, z8.d, z12.d\n\t" \
 ALD "z1.d, p0/z, [%[a], #1, mul vl]\n\t" \
 "fmopa za1.d, p0/m, p0/m, z8.d, z13.d\n\t" \
 "ld1d z4.d, p0/z, [%[b]]\n\t" \
 "fmopa za2.d, p0/m, p0/m, z9.d, z12.d\n\t" \
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t" \
 "fmopa za3.d, p0/m, p0/m, z9.d, z13.d\n\t" \
 "add %[a], %[a], #128\n\t" \
 "add %[b], %[b], #" #BS "\n\t" \
 "prfm pldl2keep, [%[pc]]\n\t" \
 "prfm pldl2keep, [%[pc], #64]\n\t" \
 "prfm pldl2keep, [%[pc], #" #C8 "]\n\t" \
 "prfm pldl2keep, [%[pc], #" #C864 "]\n\t" \
 "add %[pc], %[pc], #" #CS "\n\t" \
 ALD "z8.d, p0/z, [%[a]]\n\t" \
 "fmopa za0.d, p0/m, p0/m, z0.d, z4.d\n\t" \
 ALD "z9.d, p0/z, [%[a], #1, mul vl]\n\t" \
 "fmopa za1.d, p0/m, p0/m, z0.d, z5.d\n\t" \
 "ld1d z12.d, p0/z, [%[b]]\n\t" \
 "fmopa za2.d, p0/m, p0/m, z1.d, z4.d\n\t" \
 "ld1d z13.d, p0/z, [%[b], #1, mul vl]\n\t" \
 "fmopa za3.d, p0/m, p0/m, z1.d, z5.d\n\t" \
 "add %[a], %[a], #128\n\t" \
 "add %[b], %[b], #" #BS "\n\t" \
 ALD "z0.d, p0/z, [%[a]]\n\t" \
 "fmopa za0.d, p0/m, p0/m, z8.d, z12.d\n\t" \
 ALD "z1.d, p0/z, [%[a], #1, mul vl]\n\t" \
 "fmopa za1.d, p0/m, p0/m, z8.d, z13.d\n\t" \
 "ld1d z4.d, p0/z, [%[b]]\n\t" \
 "fmopa za2.d, p0/m, p0/m, z9.d, z12.d\n\t" \
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t" \
 "fmopa za3.d, p0/m, p0/m, z9.d, z13.d\n\t" \
 "add %[a], %[a], #128\n\t" \
 "add %[b], %[b], #" #BS "\n\t" \
 "prfm pldl2keep, [%[ap]]\n\t" \
 "prfm pldl2keep, [%[bp]]\n\t" \
 "add %[ap], %[ap], #64\n\t" \
 "add %[bp], %[bp], #64\n\t" \
 ALD "z8.d, p0/z, [%[a]]\n\t" \
 "fmopa za0.d, p0/m, p0/m, z0.d, z4.d\n\t" \
 ALD "z9.d, p0/z, [%[a], #1, mul vl]\n\t" \
 "fmopa za1.d, p0/m, p0/m, z0.d, z5.d\n\t" \
 "ld1d z12.d, p0/z, [%[b]]\n\t" \
 "fmopa za2.d, p0/m, p0/m, z1.d, z4.d\n\t" \
 "ld1d z13.d, p0/z, [%[b], #1, mul vl]\n\t" \
 "fmopa za3.d, p0/m, p0/m, z1.d, z5.d\n\t" \
 "add %[a], %[a], #128\n\t" \
 "add %[b], %[b], #" #BS "\n\t"

#define DDSET0_S_NEXT(BS, ALD) \
 ALD "z0.d, p0/z, [%[a]]\n\t" \
 "fmopa za0.d, p0/m, p0/m, z8.d, z12.d\n\t" \
 ALD "z1.d, p0/z, [%[a], #1, mul vl]\n\t" \
 "fmopa za1.d, p0/m, p0/m, z8.d, z13.d\n\t" \
 "ld1d z4.d, p0/z, [%[b]]\n\t" \
 "fmopa za2.d, p0/m, p0/m, z9.d, z12.d\n\t" \
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t" \
 "fmopa za3.d, p0/m, p0/m, z9.d, z13.d\n\t" \
 "add %[a], %[a], #128\n\t" \
 "add %[b], %[b], #" #BS "\n\t"

#define DDPURE0_P_NEXT(BS, ALD) \
 ALD "z0.d, p0/z, [%[a]]\n\t" \
 "fmopa za0.d, p0/m, p0/m, z8.d, z12.d\n\t" \
 ALD "z1.d, p0/z, [%[a], #1, mul vl]\n\t" \
 "fmopa za1.d, p0/m, p0/m, z8.d, z13.d\n\t" \
 "ld1d z4.d, p0/z, [%[b]]\n\t" \
 "fmopa za2.d, p0/m, p0/m, z9.d, z12.d\n\t" \
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t" \
 "fmopa za3.d, p0/m, p0/m, z9.d, z13.d\n\t" \
 "add %[a], %[a], #128\n\t" \
 "add %[b], %[b], #" #BS "\n\t"

#define DDSET0_S_LAST \
 "fmopa za0.d, p0/m, p0/m, z8.d, z12.d\n\t" \
 "fmopa za1.d, p0/m, p0/m, z8.d, z13.d\n\t" \
 "fmopa za2.d, p0/m, p0/m, z9.d, z12.d\n\t" \
 "fmopa za3.d, p0/m, p0/m, z9.d, z13.d\n\t"

#define DDPURE0_P_LAST \
 "fmopa za0.d, p0/m, p0/m, z8.d, z12.d\n\t" \
 "fmopa za1.d, p0/m, p0/m, z8.d, z13.d\n\t" \
 "fmopa za2.d, p0/m, p0/m, z9.d, z12.d\n\t" \
 "fmopa za3.d, p0/m, p0/m, z9.d, z13.d\n\t"

#define DDSET0_S_TAIL(CS) \
 "st1d z22.d, p0, [%[cb], #1, mul vl]\n\t" \
 "add %[ct], %[ct], #" #CS "\n\t" \
 "add %[cb], %[cb], #" #CS "\n\t"

#define DDPURE0_P_TAIL(CS) \
 "add %[ct], %[ct], #" #CS "\n\t" \
 "add %[cb], %[cb], #" #CS "\n\t"

#define DDSET4_S_BODY(BS, CS, C8, C864, ALD) \
 "mova z16.d, p0/m, za0h.d[w12, 0]\n\t" \
 "mova z18.d, p0/m, za1h.d[w12, 0]\n\t" \
 "mova z20.d, p0/m, za2h.d[w12, 0]\n\t" \
 "mova z22.d, p0/m, za3h.d[w12, 0]\n\t" \
 ALD "z8.d, p0/z, [%[a]]\n\t" \
 "fmopa za4.d, p0/m, p0/m, z0.d, z4.d\n\t" \
 ALD "z9.d, p0/z, [%[a], #1, mul vl]\n\t" \
 "fmopa za5.d, p0/m, p0/m, z0.d, z5.d\n\t" \
 "ld1d z12.d, p0/z, [%[b]]\n\t" \
 "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n\t" \
 "ld1d z13.d, p0/z, [%[b], #1, mul vl]\n\t" \
 "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n\t" \
 "add %[a], %[a], #128\n\t" \
 "add %[b], %[b], #" #BS "\n\t" \
 ALD "z0.d, p0/z, [%[a]]\n\t" \
 "fmopa za4.d, p0/m, p0/m, z8.d, z12.d\n\t" \
 ALD "z1.d, p0/z, [%[a], #1, mul vl]\n\t" \
 "fmopa za5.d, p0/m, p0/m, z8.d, z13.d\n\t" \
 "ld1d z4.d, p0/z, [%[b]]\n\t" \
 "fmopa za6.d, p0/m, p0/m, z9.d, z12.d\n\t" \
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t" \
 "fmopa za7.d, p0/m, p0/m, z9.d, z13.d\n\t" \
 "add %[a], %[a], #128\n\t" \
 "add %[b], %[b], #" #BS "\n\t" \
 "st1d z16.d, p0, [%[ct], #0, mul vl]\n\t" \
 "prfm pldl2keep, [%[ap]]\n\t" \
 "prfm pldl2keep, [%[bp]]\n\t" \
 "add %[ap], %[ap], #64\n\t" \
 "add %[bp], %[bp], #64\n\t" \
 ALD "z8.d, p0/z, [%[a]]\n\t" \
 "fmopa za4.d, p0/m, p0/m, z0.d, z4.d\n\t" \
 ALD "z9.d, p0/z, [%[a], #1, mul vl]\n\t" \
 "fmopa za5.d, p0/m, p0/m, z0.d, z5.d\n\t" \
 "ld1d z12.d, p0/z, [%[b]]\n\t" \
 "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n\t" \
 "ld1d z13.d, p0/z, [%[b], #1, mul vl]\n\t" \
 "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n\t" \
 "add %[a], %[a], #128\n\t" \
 "add %[b], %[b], #" #BS "\n\t" \
 ALD "z0.d, p0/z, [%[a]]\n\t" \
 "fmopa za4.d, p0/m, p0/m, z8.d, z12.d\n\t" \
 ALD "z1.d, p0/z, [%[a], #1, mul vl]\n\t" \
 "fmopa za5.d, p0/m, p0/m, z8.d, z13.d\n\t" \
 "ld1d z4.d, p0/z, [%[b]]\n\t" \
 "fmopa za6.d, p0/m, p0/m, z9.d, z12.d\n\t" \
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t" \
 "fmopa za7.d, p0/m, p0/m, z9.d, z13.d\n\t" \
 "add %[a], %[a], #128\n\t" \
 "add %[b], %[b], #" #BS "\n\t" \
 "st1d z18.d, p0, [%[ct], #1, mul vl]\n\t" \
 "prfm pldl2keep, [%[pc]]\n\t" \
 "prfm pldl2keep, [%[pc], #64]\n\t" \
 "prfm pldl2keep, [%[pc], #" #C8 "]\n\t" \
 "prfm pldl2keep, [%[pc], #" #C864 "]\n\t" \
 "add %[pc], %[pc], #" #CS "\n\t" \
 ALD "z8.d, p0/z, [%[a]]\n\t" \
 "fmopa za4.d, p0/m, p0/m, z0.d, z4.d\n\t" \
 ALD "z9.d, p0/z, [%[a], #1, mul vl]\n\t" \
 "fmopa za5.d, p0/m, p0/m, z0.d, z5.d\n\t" \
 "ld1d z12.d, p0/z, [%[b]]\n\t" \
 "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n\t" \
 "ld1d z13.d, p0/z, [%[b], #1, mul vl]\n\t" \
 "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n\t" \
 "add %[a], %[a], #128\n\t" \
 "add %[b], %[b], #" #BS "\n\t" \
 ALD "z0.d, p0/z, [%[a]]\n\t" \
 "fmopa za4.d, p0/m, p0/m, z8.d, z12.d\n\t" \
 ALD "z1.d, p0/z, [%[a], #1, mul vl]\n\t" \
 "fmopa za5.d, p0/m, p0/m, z8.d, z13.d\n\t" \
 "ld1d z4.d, p0/z, [%[b]]\n\t" \
 "fmopa za6.d, p0/m, p0/m, z9.d, z12.d\n\t" \
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t" \
 "fmopa za7.d, p0/m, p0/m, z9.d, z13.d\n\t" \
 "add %[a], %[a], #128\n\t" \
 "add %[b], %[b], #" #BS "\n\t" \
 "st1d z20.d, p0, [%[cb], #0, mul vl]\n\t" \
 "prfm pldl2keep, [%[ap]]\n\t" \
 "prfm pldl2keep, [%[bp]]\n\t" \
 "add %[ap], %[ap], #64\n\t" \
 "add %[bp], %[bp], #64\n\t" \
 ALD "z8.d, p0/z, [%[a]]\n\t" \
 "fmopa za4.d, p0/m, p0/m, z0.d, z4.d\n\t" \
 ALD "z9.d, p0/z, [%[a], #1, mul vl]\n\t" \
 "fmopa za5.d, p0/m, p0/m, z0.d, z5.d\n\t" \
 "ld1d z12.d, p0/z, [%[b]]\n\t" \
 "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n\t" \
 "ld1d z13.d, p0/z, [%[b], #1, mul vl]\n\t" \
 "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n\t" \
 "add %[a], %[a], #128\n\t" \
 "add %[b], %[b], #" #BS "\n\t"

#define DDPURE4_P_BODY(BS, CS, C8, C864, ALD) \
 ALD "z8.d, p0/z, [%[a]]\n\t" \
 "fmopa za4.d, p0/m, p0/m, z0.d, z4.d\n\t" \
 ALD "z9.d, p0/z, [%[a], #1, mul vl]\n\t" \
 "fmopa za5.d, p0/m, p0/m, z0.d, z5.d\n\t" \
 "ld1d z12.d, p0/z, [%[b]]\n\t" \
 "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n\t" \
 "ld1d z13.d, p0/z, [%[b], #1, mul vl]\n\t" \
 "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n\t" \
 "add %[a], %[a], #128\n\t" \
 "add %[b], %[b], #" #BS "\n\t" \
 ALD "z0.d, p0/z, [%[a]]\n\t" \
 "fmopa za4.d, p0/m, p0/m, z8.d, z12.d\n\t" \
 ALD "z1.d, p0/z, [%[a], #1, mul vl]\n\t" \
 "fmopa za5.d, p0/m, p0/m, z8.d, z13.d\n\t" \
 "ld1d z4.d, p0/z, [%[b]]\n\t" \
 "fmopa za6.d, p0/m, p0/m, z9.d, z12.d\n\t" \
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t" \
 "fmopa za7.d, p0/m, p0/m, z9.d, z13.d\n\t" \
 "add %[a], %[a], #128\n\t" \
 "add %[b], %[b], #" #BS "\n\t" \
 "prfm pldl2keep, [%[ap]]\n\t" \
 "prfm pldl2keep, [%[bp]]\n\t" \
 "add %[ap], %[ap], #64\n\t" \
 "add %[bp], %[bp], #64\n\t" \
 ALD "z8.d, p0/z, [%[a]]\n\t" \
 "fmopa za4.d, p0/m, p0/m, z0.d, z4.d\n\t" \
 ALD "z9.d, p0/z, [%[a], #1, mul vl]\n\t" \
 "fmopa za5.d, p0/m, p0/m, z0.d, z5.d\n\t" \
 "ld1d z12.d, p0/z, [%[b]]\n\t" \
 "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n\t" \
 "ld1d z13.d, p0/z, [%[b], #1, mul vl]\n\t" \
 "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n\t" \
 "add %[a], %[a], #128\n\t" \
 "add %[b], %[b], #" #BS "\n\t" \
 ALD "z0.d, p0/z, [%[a]]\n\t" \
 "fmopa za4.d, p0/m, p0/m, z8.d, z12.d\n\t" \
 ALD "z1.d, p0/z, [%[a], #1, mul vl]\n\t" \
 "fmopa za5.d, p0/m, p0/m, z8.d, z13.d\n\t" \
 "ld1d z4.d, p0/z, [%[b]]\n\t" \
 "fmopa za6.d, p0/m, p0/m, z9.d, z12.d\n\t" \
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t" \
 "fmopa za7.d, p0/m, p0/m, z9.d, z13.d\n\t" \
 "add %[a], %[a], #128\n\t" \
 "add %[b], %[b], #" #BS "\n\t" \
 "prfm pldl2keep, [%[pc]]\n\t" \
 "prfm pldl2keep, [%[pc], #64]\n\t" \
 "prfm pldl2keep, [%[pc], #" #C8 "]\n\t" \
 "prfm pldl2keep, [%[pc], #" #C864 "]\n\t" \
 "add %[pc], %[pc], #" #CS "\n\t" \
 ALD "z8.d, p0/z, [%[a]]\n\t" \
 "fmopa za4.d, p0/m, p0/m, z0.d, z4.d\n\t" \
 ALD "z9.d, p0/z, [%[a], #1, mul vl]\n\t" \
 "fmopa za5.d, p0/m, p0/m, z0.d, z5.d\n\t" \
 "ld1d z12.d, p0/z, [%[b]]\n\t" \
 "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n\t" \
 "ld1d z13.d, p0/z, [%[b], #1, mul vl]\n\t" \
 "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n\t" \
 "add %[a], %[a], #128\n\t" \
 "add %[b], %[b], #" #BS "\n\t" \
 ALD "z0.d, p0/z, [%[a]]\n\t" \
 "fmopa za4.d, p0/m, p0/m, z8.d, z12.d\n\t" \
 ALD "z1.d, p0/z, [%[a], #1, mul vl]\n\t" \
 "fmopa za5.d, p0/m, p0/m, z8.d, z13.d\n\t" \
 "ld1d z4.d, p0/z, [%[b]]\n\t" \
 "fmopa za6.d, p0/m, p0/m, z9.d, z12.d\n\t" \
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t" \
 "fmopa za7.d, p0/m, p0/m, z9.d, z13.d\n\t" \
 "add %[a], %[a], #128\n\t" \
 "add %[b], %[b], #" #BS "\n\t" \
 "prfm pldl2keep, [%[ap]]\n\t" \
 "prfm pldl2keep, [%[bp]]\n\t" \
 "add %[ap], %[ap], #64\n\t" \
 "add %[bp], %[bp], #64\n\t" \
 ALD "z8.d, p0/z, [%[a]]\n\t" \
 "fmopa za4.d, p0/m, p0/m, z0.d, z4.d\n\t" \
 ALD "z9.d, p0/z, [%[a], #1, mul vl]\n\t" \
 "fmopa za5.d, p0/m, p0/m, z0.d, z5.d\n\t" \
 "ld1d z12.d, p0/z, [%[b]]\n\t" \
 "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n\t" \
 "ld1d z13.d, p0/z, [%[b], #1, mul vl]\n\t" \
 "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n\t" \
 "add %[a], %[a], #128\n\t" \
 "add %[b], %[b], #" #BS "\n\t"

#define DDSET4_S_NEXT(BS, ALD) \
 ALD "z0.d, p0/z, [%[a]]\n\t" \
 "fmopa za4.d, p0/m, p0/m, z8.d, z12.d\n\t" \
 ALD "z1.d, p0/z, [%[a], #1, mul vl]\n\t" \
 "fmopa za5.d, p0/m, p0/m, z8.d, z13.d\n\t" \
 "ld1d z4.d, p0/z, [%[b]]\n\t" \
 "fmopa za6.d, p0/m, p0/m, z9.d, z12.d\n\t" \
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t" \
 "fmopa za7.d, p0/m, p0/m, z9.d, z13.d\n\t" \
 "add %[a], %[a], #128\n\t" \
 "add %[b], %[b], #" #BS "\n\t"

#define DDPURE4_P_NEXT(BS, ALD) \
 ALD "z0.d, p0/z, [%[a]]\n\t" \
 "fmopa za4.d, p0/m, p0/m, z8.d, z12.d\n\t" \
 ALD "z1.d, p0/z, [%[a], #1, mul vl]\n\t" \
 "fmopa za5.d, p0/m, p0/m, z8.d, z13.d\n\t" \
 "ld1d z4.d, p0/z, [%[b]]\n\t" \
 "fmopa za6.d, p0/m, p0/m, z9.d, z12.d\n\t" \
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t" \
 "fmopa za7.d, p0/m, p0/m, z9.d, z13.d\n\t" \
 "add %[a], %[a], #128\n\t" \
 "add %[b], %[b], #" #BS "\n\t"

#define DDSET4_S_LAST \
 "fmopa za4.d, p0/m, p0/m, z8.d, z12.d\n\t" \
 "fmopa za5.d, p0/m, p0/m, z8.d, z13.d\n\t" \
 "fmopa za6.d, p0/m, p0/m, z9.d, z12.d\n\t" \
 "fmopa za7.d, p0/m, p0/m, z9.d, z13.d\n\t"

#define DDPURE4_P_LAST \
 "fmopa za4.d, p0/m, p0/m, z8.d, z12.d\n\t" \
 "fmopa za5.d, p0/m, p0/m, z8.d, z13.d\n\t" \
 "fmopa za6.d, p0/m, p0/m, z9.d, z12.d\n\t" \
 "fmopa za7.d, p0/m, p0/m, z9.d, z13.d\n\t"

#define DDSET4_S_TAIL(CS) \
 "st1d z22.d, p0, [%[cb], #1, mul vl]\n\t" \
 "add %[ct], %[ct], #" #CS "\n\t" \
 "add %[cb], %[cb], #" #CS "\n\t"

#define DDPURE4_P_TAIL(CS) \
 "add %[ct], %[ct], #" #CS "\n\t" \
 "add %[cb], %[cb], #" #CS "\n\t"


static inline __attribute__((always_inline)) void dd_b0_first(const double* a, const double* b, double* c,
                                      const char* ap, const char* bp, const char* pc)
{
    char* ct = (char*)c;
    char* cb = ct + 8192;
    asm volatile(
 "ptrue p0.d\n\t"
 "zero {za0.d, za1.d, za2.d, za3.d}\n\t"
 "ld1d z0.d, p0/z, [%[a]]\n\t"
 "ld1d z1.d, p0/z, [%[a], #1, mul vl]\n\t"
 "ld1d z4.d, p0/z, [%[b]]\n\t"
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #1024\n\t"
 "mov w12, #0\n\t" DDSET0_S_BODY(1024, 1024, 8192, 8256, "ld1d ") DDSET0_S_NEXT(1024, "ld1d ") DDSET0_S_TAIL(1024)
 "mov w12, #1\n\t" DDSET0_S_BODY(1024, 1024, 8192, 8256, "ld1d ") DDSET0_S_NEXT(1024, "ld1d ") DDSET0_S_TAIL(1024)
 "mov w12, #2\n\t" DDSET0_S_BODY(1024, 1024, 8192, 8256, "ld1d ") DDSET0_S_NEXT(1024, "ld1d ") DDSET0_S_TAIL(1024)
 "mov w12, #3\n\t" DDSET0_S_BODY(1024, 1024, 8192, 8256, "ld1d ") DDSET0_S_NEXT(1024, "ld1d ") DDSET0_S_TAIL(1024)
 "mov w12, #4\n\t" DDSET0_S_BODY(1024, 1024, 8192, 8256, "ld1d ") DDSET0_S_NEXT(1024, "ld1d ") DDSET0_S_TAIL(1024)
 "mov w12, #5\n\t" DDSET0_S_BODY(1024, 1024, 8192, 8256, "ld1d ") DDSET0_S_NEXT(1024, "ld1d ") DDSET0_S_TAIL(1024)
 "mov w12, #6\n\t" DDSET0_S_BODY(1024, 1024, 8192, 8256, "ld1d ") DDSET0_S_NEXT(1024, "ld1d ") DDSET0_S_TAIL(1024)
 "mov w12, #7\n\t" DDSET0_S_BODY(1024, 1024, 8192, 8256, "ld1d ") DDSET0_S_LAST DDSET0_S_TAIL(1024)
 : [a] "+&r"(a),[b] "+&r"(b),[ct] "+&r"(ct),[cb] "+&r"(cb),[ap] "+&r"(ap),[bp] "+&r"(bp),[pc] "+&r"(pc)
 :: "memory","cc","x12",VCLOB,PCLOB);
}

static inline __attribute__((always_inline)) void dd_b0_pure(const double* a, const double* b, double* c,
                                      const char* ap, const char* bp, const char* pc)
{
    char* ct = (char*)c;
    char* cb = ct + 8192;
    asm volatile(
 "ptrue p0.d\n\t"
 "ld1d z0.d, p0/z, [%[a]]\n\t"
 "ld1d z1.d, p0/z, [%[a], #1, mul vl]\n\t"
 "ld1d z4.d, p0/z, [%[b]]\n\t"
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #1024\n\t"
 "mov w12, #0\n\t" DDPURE0_P_BODY(1024, 1024, 8192, 8256, "ld1d ") DDPURE0_P_NEXT(1024, "ld1d ") DDPURE0_P_TAIL(1024)
 "mov w12, #1\n\t" DDPURE0_P_BODY(1024, 1024, 8192, 8256, "ld1d ") DDPURE0_P_NEXT(1024, "ld1d ") DDPURE0_P_TAIL(1024)
 "mov w12, #2\n\t" DDPURE0_P_BODY(1024, 1024, 8192, 8256, "ld1d ") DDPURE0_P_NEXT(1024, "ld1d ") DDPURE0_P_TAIL(1024)
 "mov w12, #3\n\t" DDPURE0_P_BODY(1024, 1024, 8192, 8256, "ld1d ") DDPURE0_P_NEXT(1024, "ld1d ") DDPURE0_P_TAIL(1024)
 "mov w12, #4\n\t" DDPURE0_P_BODY(1024, 1024, 8192, 8256, "ld1d ") DDPURE0_P_NEXT(1024, "ld1d ") DDPURE0_P_TAIL(1024)
 "mov w12, #5\n\t" DDPURE0_P_BODY(1024, 1024, 8192, 8256, "ld1d ") DDPURE0_P_NEXT(1024, "ld1d ") DDPURE0_P_TAIL(1024)
 "mov w12, #6\n\t" DDPURE0_P_BODY(1024, 1024, 8192, 8256, "ld1d ") DDPURE0_P_NEXT(1024, "ld1d ") DDPURE0_P_TAIL(1024)
 "mov w12, #7\n\t" DDPURE0_P_BODY(1024, 1024, 8192, 8256, "ld1d ") DDPURE0_P_LAST DDPURE0_P_TAIL(1024)
 : [a] "+&r"(a),[b] "+&r"(b),[ct] "+&r"(ct),[cb] "+&r"(cb),[ap] "+&r"(ap),[bp] "+&r"(bp),[pc] "+&r"(pc)
 :: "memory","cc","x12",VCLOB,PCLOB);
}

static inline __attribute__((always_inline)) void dd_b4_zlast(const double* a, const double* b, double* c,
                                      const char* ap, const char* bp, const char* pc)
{
    char* ct = (char*)c;
    char* cb = ct + 8192;
    asm volatile(
 "ptrue p0.d\n\t"
 "zero {za4.d, za5.d, za6.d, za7.d}\n\t"
 "ld1d z0.d, p0/z, [%[a]]\n\t"
 "ld1d z1.d, p0/z, [%[a], #1, mul vl]\n\t"
 "ld1d z4.d, p0/z, [%[b]]\n\t"
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #1024\n\t"
 "mov w12, #0\n\t" DDSET4_S_BODY(1024, 1024, 8192, 8256, "ld1d ") DDSET4_S_NEXT(1024, "ld1d ") DDSET4_S_TAIL(1024)
 "mov w12, #1\n\t" DDSET4_S_BODY(1024, 1024, 8192, 8256, "ld1d ") DDSET4_S_NEXT(1024, "ld1d ") DDSET4_S_TAIL(1024)
 "mov w12, #2\n\t" DDSET4_S_BODY(1024, 1024, 8192, 8256, "ld1d ") DDSET4_S_NEXT(1024, "ld1d ") DDSET4_S_TAIL(1024)
 "mov w12, #3\n\t" DDSET4_S_BODY(1024, 1024, 8192, 8256, "ld1d ") DDSET4_S_NEXT(1024, "ld1d ") DDSET4_S_TAIL(1024)
 "mov w12, #4\n\t" DDSET4_S_BODY(1024, 1024, 8192, 8256, "ld1d ") DDSET4_S_NEXT(1024, "ld1d ") DDSET4_S_TAIL(1024)
 "mov w12, #5\n\t" DDSET4_S_BODY(1024, 1024, 8192, 8256, "ld1d ") DDSET4_S_NEXT(1024, "ld1d ") DDSET4_S_TAIL(1024)
 "mov w12, #6\n\t" DDSET4_S_BODY(1024, 1024, 8192, 8256, "ld1d ") DDSET4_S_NEXT(1024, "ld1d ") DDSET4_S_TAIL(1024)
 "mov w12, #7\n\t" DDSET4_S_BODY(1024, 1024, 8192, 8256, "ld1d ") DDSET4_S_LAST DDSET4_S_TAIL(1024)
 : [a] "+&r"(a),[b] "+&r"(b),[ct] "+&r"(ct),[cb] "+&r"(cb),[ap] "+&r"(ap),[bp] "+&r"(bp),[pc] "+&r"(pc)
 :: "memory","cc","x12",VCLOB,PCLOB);
}

static inline __attribute__((always_inline)) void dd_b4_zpure(const double* a, const double* b, double* c,
                                      const char* ap, const char* bp, const char* pc)
{
    char* ct = (char*)c;
    char* cb = ct + 8192;
    asm volatile(
 "ptrue p0.d\n\t"
 "zero {za4.d, za5.d, za6.d, za7.d}\n\t"
 "ld1d z0.d, p0/z, [%[a]]\n\t"
 "ld1d z1.d, p0/z, [%[a], #1, mul vl]\n\t"
 "ld1d z4.d, p0/z, [%[b]]\n\t"
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #1024\n\t"
 "mov w12, #0\n\t" DDPURE4_P_BODY(1024, 1024, 8192, 8256, "ld1d ") DDPURE4_P_NEXT(1024, "ld1d ") DDPURE4_P_TAIL(1024)
 "mov w12, #1\n\t" DDPURE4_P_BODY(1024, 1024, 8192, 8256, "ld1d ") DDPURE4_P_NEXT(1024, "ld1d ") DDPURE4_P_TAIL(1024)
 "mov w12, #2\n\t" DDPURE4_P_BODY(1024, 1024, 8192, 8256, "ld1d ") DDPURE4_P_NEXT(1024, "ld1d ") DDPURE4_P_TAIL(1024)
 "mov w12, #3\n\t" DDPURE4_P_BODY(1024, 1024, 8192, 8256, "ld1d ") DDPURE4_P_NEXT(1024, "ld1d ") DDPURE4_P_TAIL(1024)
 "mov w12, #4\n\t" DDPURE4_P_BODY(1024, 1024, 8192, 8256, "ld1d ") DDPURE4_P_NEXT(1024, "ld1d ") DDPURE4_P_TAIL(1024)
 "mov w12, #5\n\t" DDPURE4_P_BODY(1024, 1024, 8192, 8256, "ld1d ") DDPURE4_P_NEXT(1024, "ld1d ") DDPURE4_P_TAIL(1024)
 "mov w12, #6\n\t" DDPURE4_P_BODY(1024, 1024, 8192, 8256, "ld1d ") DDPURE4_P_NEXT(1024, "ld1d ") DDPURE4_P_TAIL(1024)
 "mov w12, #7\n\t" DDPURE4_P_BODY(1024, 1024, 8192, 8256, "ld1d ") DDPURE4_P_LAST DDPURE4_P_TAIL(1024)
 : [a] "+&r"(a),[b] "+&r"(b),[ct] "+&r"(ct),[cb] "+&r"(cb),[ap] "+&r"(ap),[bp] "+&r"(bp),[pc] "+&r"(pc)
 :: "memory","cc","x12",VCLOB,PCLOB);
}

static inline __attribute__((always_inline)) void dd_b4_pure(const double* a, const double* b, double* c,
                                      const char* ap, const char* bp, const char* pc)
{
    char* ct = (char*)c;
    char* cb = ct + 8192;
    asm volatile(
 "ptrue p0.d\n\t"
 "ld1d z0.d, p0/z, [%[a]]\n\t"
 "ld1d z1.d, p0/z, [%[a], #1, mul vl]\n\t"
 "ld1d z4.d, p0/z, [%[b]]\n\t"
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #1024\n\t"
 "mov w12, #0\n\t" DDPURE4_P_BODY(1024, 1024, 8192, 8256, "ld1d ") DDPURE4_P_NEXT(1024, "ld1d ") DDPURE4_P_TAIL(1024)
 "mov w12, #1\n\t" DDPURE4_P_BODY(1024, 1024, 8192, 8256, "ld1d ") DDPURE4_P_NEXT(1024, "ld1d ") DDPURE4_P_TAIL(1024)
 "mov w12, #2\n\t" DDPURE4_P_BODY(1024, 1024, 8192, 8256, "ld1d ") DDPURE4_P_NEXT(1024, "ld1d ") DDPURE4_P_TAIL(1024)
 "mov w12, #3\n\t" DDPURE4_P_BODY(1024, 1024, 8192, 8256, "ld1d ") DDPURE4_P_NEXT(1024, "ld1d ") DDPURE4_P_TAIL(1024)
 "mov w12, #4\n\t" DDPURE4_P_BODY(1024, 1024, 8192, 8256, "ld1d ") DDPURE4_P_NEXT(1024, "ld1d ") DDPURE4_P_TAIL(1024)
 "mov w12, #5\n\t" DDPURE4_P_BODY(1024, 1024, 8192, 8256, "ld1d ") DDPURE4_P_NEXT(1024, "ld1d ") DDPURE4_P_TAIL(1024)
 "mov w12, #6\n\t" DDPURE4_P_BODY(1024, 1024, 8192, 8256, "ld1d ") DDPURE4_P_NEXT(1024, "ld1d ") DDPURE4_P_TAIL(1024)
 "mov w12, #7\n\t" DDPURE4_P_BODY(1024, 1024, 8192, 8256, "ld1d ") DDPURE4_P_LAST DDPURE4_P_TAIL(1024)
 : [a] "+&r"(a),[b] "+&r"(b),[ct] "+&r"(ct),[cb] "+&r"(cb),[ap] "+&r"(ap),[bp] "+&r"(bp),[pc] "+&r"(pc)
 :: "memory","cc","x12",VCLOB,PCLOB);
}

static inline __attribute__((always_inline)) void dd_b4_last(const double* a, const double* b, double* c,
                                      const char* ap, const char* bp, const char* pc)
{
    char* ct = (char*)c;
    char* cb = ct + 8192;
    asm volatile(
 "ptrue p0.d\n\t"
 "ld1d z0.d, p0/z, [%[a]]\n\t"
 "ld1d z1.d, p0/z, [%[a], #1, mul vl]\n\t"
 "ld1d z4.d, p0/z, [%[b]]\n\t"
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #1024\n\t"
 "mov w12, #0\n\t" DDSET4_S_BODY(1024, 1024, 8192, 8256, "ld1d ") DDSET4_S_NEXT(1024, "ld1d ") DDSET4_S_TAIL(1024)
 "mov w12, #1\n\t" DDSET4_S_BODY(1024, 1024, 8192, 8256, "ld1d ") DDSET4_S_NEXT(1024, "ld1d ") DDSET4_S_TAIL(1024)
 "mov w12, #2\n\t" DDSET4_S_BODY(1024, 1024, 8192, 8256, "ld1d ") DDSET4_S_NEXT(1024, "ld1d ") DDSET4_S_TAIL(1024)
 "mov w12, #3\n\t" DDSET4_S_BODY(1024, 1024, 8192, 8256, "ld1d ") DDSET4_S_NEXT(1024, "ld1d ") DDSET4_S_TAIL(1024)
 "mov w12, #4\n\t" DDSET4_S_BODY(1024, 1024, 8192, 8256, "ld1d ") DDSET4_S_NEXT(1024, "ld1d ") DDSET4_S_TAIL(1024)
 "mov w12, #5\n\t" DDSET4_S_BODY(1024, 1024, 8192, 8256, "ld1d ") DDSET4_S_NEXT(1024, "ld1d ") DDSET4_S_TAIL(1024)
 "mov w12, #6\n\t" DDSET4_S_BODY(1024, 1024, 8192, 8256, "ld1d ") DDSET4_S_NEXT(1024, "ld1d ") DDSET4_S_TAIL(1024)
 "mov w12, #7\n\t" DDSET4_S_BODY(1024, 1024, 8192, 8256, "ld1d ") DDSET4_S_LAST DDSET4_S_TAIL(1024)
 : [a] "+&r"(a),[b] "+&r"(b),[ct] "+&r"(ct),[cb] "+&r"(cb),[ap] "+&r"(ap),[bp] "+&r"(bp),[pc] "+&r"(pc)
 :: "memory","cc","x12",VCLOB,PCLOB);
}

static inline __attribute__((always_inline)) void dd_b0_rem(const double* a, const double* b, long k)
{
    asm volatile(
 "ptrue p0.d\n\t"
 "ld1d z0.d, p0/z, [%[a]]\n\t"
 "ld1d z1.d, p0/z, [%[a], #1, mul vl]\n\t"
 "ld1d z4.d, p0/z, [%[b]]\n\t"
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #1024\n\t"
 "1:\n\t"
 "ld1d z8.d, p0/z, [%[a]]\n\t"
 "fmopa za0.d, p0/m, p0/m, z0.d, z4.d\n\t"
 "ld1d z9.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za1.d, p0/m, p0/m, z0.d, z5.d\n\t"
 "ld1d z12.d, p0/z, [%[b]]\n\t"
 "fmopa za2.d, p0/m, p0/m, z1.d, z4.d\n\t"
 "ld1d z13.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za3.d, p0/m, p0/m, z1.d, z5.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #1024\n\t"
 "ld1d z0.d, p0/z, [%[a]]\n\t"
 "fmopa za0.d, p0/m, p0/m, z8.d, z12.d\n\t"
 "ld1d z1.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za1.d, p0/m, p0/m, z8.d, z13.d\n\t"
 "ld1d z4.d, p0/z, [%[b]]\n\t"
 "fmopa za2.d, p0/m, p0/m, z9.d, z12.d\n\t"
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za3.d, p0/m, p0/m, z9.d, z13.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #1024\n\t"
 "subs %[k], %[k], #2\n\t"
 "b.ne 1b\n\t"
 : [a] "+&r"(a),[b] "+&r"(b),[k] "+&r"(k)
 :: "memory","cc",VCLOB,PCLOB);
}

static inline __attribute__((always_inline)) void dd_b4_rem(const double* a, const double* b, long k)
{
    asm volatile(
 "ptrue p0.d\n\t"
 "ld1d z0.d, p0/z, [%[a]]\n\t"
 "ld1d z1.d, p0/z, [%[a], #1, mul vl]\n\t"
 "ld1d z4.d, p0/z, [%[b]]\n\t"
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #1024\n\t"
 "1:\n\t"
 "ld1d z8.d, p0/z, [%[a]]\n\t"
 "fmopa za4.d, p0/m, p0/m, z0.d, z4.d\n\t"
 "ld1d z9.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za5.d, p0/m, p0/m, z0.d, z5.d\n\t"
 "ld1d z12.d, p0/z, [%[b]]\n\t"
 "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n\t"
 "ld1d z13.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #1024\n\t"
 "ld1d z0.d, p0/z, [%[a]]\n\t"
 "fmopa za4.d, p0/m, p0/m, z8.d, z12.d\n\t"
 "ld1d z1.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za5.d, p0/m, p0/m, z8.d, z13.d\n\t"
 "ld1d z4.d, p0/z, [%[b]]\n\t"
 "fmopa za6.d, p0/m, p0/m, z9.d, z12.d\n\t"
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za7.d, p0/m, p0/m, z9.d, z13.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #1024\n\t"
 "subs %[k], %[k], #2\n\t"
 "b.ne 1b\n\t"
 : [a] "+&r"(a),[b] "+&r"(b),[k] "+&r"(k)
 :: "memory","cc",VCLOB,PCLOB);
}

static void dd_flush_b0_set(double* c, long cs)
{
    char* ct = (char*)c;
    char* cb = ct + 8 * cs;
    asm volatile(
 "ptrue p0.d\n\t"
 "mov w12, #0\n\t"
 "1:\n\t"
 "mova z16.d, p0/m, za0h.d[w12, 0]\n\t"
 "mova z18.d, p0/m, za1h.d[w12, 0]\n\t"
 "mova z20.d, p0/m, za2h.d[w12, 0]\n\t"
 "mova z22.d, p0/m, za3h.d[w12, 0]\n\t"
 "st1d z16.d, p0, [%[ct], #0, mul vl]\n\t"
 "st1d z18.d, p0, [%[ct], #1, mul vl]\n\t"
 "st1d z20.d, p0, [%[cb], #0, mul vl]\n\t"
 "st1d z22.d, p0, [%[cb], #1, mul vl]\n\t"
 "add %[ct], %[ct], %[cs]\n\t"
 "add %[cb], %[cb], %[cs]\n\t"
 "add w12, w12, #1\n\t"
 "cmp w12, #8\n\t"
 "b.ne 1b\n\t"
 : [ct] "+&r"(ct),[cb] "+&r"(cb)
 : [cs] "r"(cs)
 : "memory","cc","x12",VCLOB,PCLOB);
}

static void dd_flush_b4_set(double* c, long cs)
{
    char* ct = (char*)c;
    char* cb = ct + 8 * cs;
    asm volatile(
 "ptrue p0.d\n\t"
 "mov w12, #0\n\t"
 "1:\n\t"
 "mova z16.d, p0/m, za4h.d[w12, 0]\n\t"
 "mova z18.d, p0/m, za5h.d[w12, 0]\n\t"
 "mova z20.d, p0/m, za6h.d[w12, 0]\n\t"
 "mova z22.d, p0/m, za7h.d[w12, 0]\n\t"
 "st1d z16.d, p0, [%[ct], #0, mul vl]\n\t"
 "st1d z18.d, p0, [%[ct], #1, mul vl]\n\t"
 "st1d z20.d, p0, [%[cb], #0, mul vl]\n\t"
 "st1d z22.d, p0, [%[cb], #1, mul vl]\n\t"
 "add %[ct], %[ct], %[cs]\n\t"
 "add %[cb], %[cb], %[cs]\n\t"
 "add w12, w12, #1\n\t"
 "cmp w12, #8\n\t"
 "b.ne 1b\n\t"
 : [ct] "+&r"(ct),[cb] "+&r"(cb)
 : [cs] "r"(cs)
 : "memory","cc","x12",VCLOB,PCLOB);
}

// ---------------- 调度与对照 ----------------
static double* al(size_t bytes) { void* p = aligned_alloc(4096, bytes); memset(p, 0, bytes); return p; }
static char g_dum[2 << 20] __attribute__((aligned(64)));
static sw_t g_swd;
static int g_newmax = 15;
static int g_alt = 0;
static pf_t pf_dummy;   // main 里初始化：pfn 大值 + pfj=0（避免 pfn=1 的每次行末处理税，贴近生产真游标形态）
#define NCP 128L
#define CS  1024L
#define SS  16
static double *gA, *gCB1, *gCB2, *gscr, *gxpan;
static uint64_t t_b0[4], t_b4[4], n_b0[4], n_b4[4];   // 0=first 1=pure 2=rem 3=(b4:zlast/zpure/last)

static void seed(double* cb) { for (long i = 0; i < 256 * NCP; i++) cb[i] = (double)((i * 2654435761u) % 1000) / 997.0; }

// 旧路径一条 strip（生产 pend 链语义）；ret: pend 状态
static int old_strip(int s, double* cb, int pend, double** pend_c, long* pend_cols, int flush_end)
{
    long K = 16L * (s + 1);
    pf_t pf = pf_dummy;
    for (long t = 0; t < 4; t++) {
        const double* A = gA + diag_strip_off(s);
        if (s + 1 < SS) pf.p2 = (const char*)(gA + diag_strip_off(s + 1)) + t * K * 32;
        pf.pc = (const char*)(cb + 16L * s * NCP + 32 * t); pf.pcs = CS;
        sme_zero_za();
        if (pend && K >= 65)
            kloop_diag_set_compact(A, 128, 128, cb + 32 * t, CS, K, 32, &pf, gscr, *pend_c, CS, *pend_cols);
        else if (pend && K == 64)
            kloop_diag64_set_compact(A, 128, 128, cb + 32 * t, CS, 32, &pf, gscr, *pend_c, CS, *pend_cols);
        else {
            if (pend) epi_set_std(gscr, *pend_c, CS, *pend_cols);
            kloop_diag_plain_compact(A, 128, 128, cb + 32 * t, CS, K, 32, &pf);
        }
        za_dump(gscr);
        pend = 1; *pend_c = cb + 16L * s * NCP + 32 * t; *pend_cols = 32;
    }
    if (flush_end && pend) { epi_set_std(gscr, *pend_c, CS, *pend_cols); pend = 0; }
    return pend;
}

static int g_tic = 0;   // W 计时轮裸跑（避免 cyc() 开销偏置新路径），诊断轮才分桶
#define TIC(bucket, idx, call) do { if (!g_tic) { call; } else { uint64_t _t = cyc(); call; bucket[idx] += cyc() - _t; n_##bucket[idx]++; } } while (0)
#define n_t_b0 n_b0
#define n_t_b4 n_b4

// 新路径一条 strip（s>=3）；d4c 跨瓦片链
static void new_strip(int s, double* cb, double** d4c)
{
    long K = 16L * (s + 1), nch = K / 64, rem = K % 64;
    const double* As = gA + diag_strip_off(s);
    const char* dum = (const char*)gxpan;
    for (long t = 0; t < 4; t++) {
        const double* B0 = cb + 32 * t;
        const double* B4 = B0 + 16;
        double* cur = cb + 16L * s * NCP + 32 * t;
        #define AB(ck) (As + (ck) * 1024)
        #define BB0(ck) (B0 + (ck) * 64 * NCP)
        #define BB4(ck) (B4 + (ck) * 64 * NCP)
        if (nch == 1) {
            TIC(t_b0, 0, dd_b0_first(AB(0), BB0(0), *d4c, dum, dum, dum));
            TIC(t_b4, 3, dd_b4_zlast(AB(0), BB4(0), cur, dum, dum, dum));
        } else if (nch == 2 && rem) {
            TIC(t_b0, 0, dd_b0_first(AB(0), BB0(0), *d4c, dum, dum, dum));
            TIC(t_b0, 2, dd_b0_rem(AB(1), BB0(1), rem));
            if (g_alt) {
                dd_flush_b0_set(cur, CS);
                TIC(t_b4, 3, dd_b4_zpure(AB(0), BB4(0), gxpan, dum, dum, dum));
            } else {
                TIC(t_b4, 3, dd_b4_zlast(AB(0), BB4(0), cur, dum, dum, dum));
            }
            TIC(t_b4, 2, dd_b4_rem(AB(1), BB4(1), rem));
        } else if (nch == 2) {
            TIC(t_b0, 0, dd_b0_first(AB(0), BB0(0), *d4c, dum, dum, dum));
            TIC(t_b0, 1, dd_b0_pure(AB(1), BB0(1), gxpan, dum, dum, dum));
            TIC(t_b4, 3, dd_b4_zpure(AB(0), BB4(0), gxpan, dum, dum, dum));
            TIC(t_b4, 3, dd_b4_last(AB(1), BB4(1), cur, dum, dum, dum));
        } else if (nch == 3 && rem) {
            TIC(t_b0, 0, dd_b0_first(AB(0), BB0(0), *d4c, dum, dum, dum));
            TIC(t_b4, 3, dd_b4_zpure(AB(0), BB4(0), gxpan, dum, dum, dum));
            TIC(t_b0, 1, dd_b0_pure(AB(1), BB0(1), gxpan, dum, dum, dum));
            TIC(t_b0, 2, dd_b0_rem(AB(2), BB0(2), rem));
            TIC(t_b4, 3, dd_b4_last(AB(1), BB4(1), cur, dum, dum, dum));
            TIC(t_b4, 2, dd_b4_rem(AB(2), BB4(2), rem));
        } else if (nch == 3) {
            TIC(t_b0, 0, dd_b0_first(AB(0), BB0(0), *d4c, dum, dum, dum));
            TIC(t_b4, 3, dd_b4_zpure(AB(0), BB4(0), gxpan, dum, dum, dum));
            TIC(t_b0, 1, dd_b0_pure(AB(1), BB0(1), gxpan, dum, dum, dum));
            TIC(t_b4, 1, dd_b4_pure(AB(1), BB4(1), gxpan, dum, dum, dum));
            TIC(t_b0, 1, dd_b0_pure(AB(2), BB0(2), gxpan, dum, dum, dum));
            TIC(t_b4, 3, dd_b4_last(AB(2), BB4(2), cur, dum, dum, dum));
        } else if (nch == 4 && rem) {
            TIC(t_b0, 0, dd_b0_first(AB(0), BB0(0), *d4c, dum, dum, dum));
            TIC(t_b4, 3, dd_b4_zpure(AB(0), BB4(0), gxpan, dum, dum, dum));
            TIC(t_b0, 1, dd_b0_pure(AB(1), BB0(1), gxpan, dum, dum, dum));
            TIC(t_b4, 1, dd_b4_pure(AB(1), BB4(1), gxpan, dum, dum, dum));
            TIC(t_b0, 1, dd_b0_pure(AB(2), BB0(2), gxpan, dum, dum, dum));
            TIC(t_b0, 2, dd_b0_rem(AB(3), BB0(3), rem));
            TIC(t_b4, 3, dd_b4_last(AB(2), BB4(2), cur, dum, dum, dum));
            TIC(t_b4, 2, dd_b4_rem(AB(3), BB4(3), rem));
        } else {
            TIC(t_b0, 0, dd_b0_first(AB(0), BB0(0), *d4c, dum, dum, dum));
            TIC(t_b4, 3, dd_b4_zpure(AB(0), BB4(0), gxpan, dum, dum, dum));
            TIC(t_b0, 1, dd_b0_pure(AB(1), BB0(1), gxpan, dum, dum, dum));
            TIC(t_b4, 1, dd_b4_pure(AB(1), BB4(1), gxpan, dum, dum, dum));
            TIC(t_b0, 1, dd_b0_pure(AB(2), BB0(2), gxpan, dum, dum, dum));
            TIC(t_b4, 1, dd_b4_pure(AB(2), BB4(2), gxpan, dum, dum, dum));
            TIC(t_b0, 1, dd_b0_pure(AB(3), BB0(3), gxpan, dum, dum, dum));
            TIC(t_b4, 3, dd_b4_last(AB(3), BB4(3), cur, dum, dum, dum));
        }
        *d4c = cur + 16;
        #undef AB
        #undef BB0
        #undef BB4
    }
}

int main(void)
{
    { const char* e = getenv("DD_NEWMAX"); if (e) g_newmax = atoi(e); }
    { const char* e = getenv("DD_ALT"); if (e) g_alt = atoi(e); }
    const double c2c = 1.55e9 / (double)cyc_hz();
    gA = al(300 * 1024);        // 对角包 272KB + 余量
    gCB1 = al(300 * 1024);      // 旧路径 cbuf（256KB + 越读余量）
    gCB2 = al(300 * 1024);      // 新路径 cbuf
    gscr = al(16 * 1024);
    gxpan = al(32 * 1024);
    for (long i = 0; i < 35000; i++) gA[i] = (double)((i * 40503u) % 913) / 911.0 - 0.5;

    pf_dummy.p = g_dum; pf_dummy.pfn = 1; pf_dummy.pfj = -64;
    pf_dummy.p2 = g_dum; pf_dummy.pc = g_dum; pf_dummy.pcs = 0; pf_dummy.sw = &g_swd;
    { const char* e = getenv("DD_UNIT"); if (e && atoi(e)) {
        // rem 内核隔离单测：za 清零 → dd_b0_rem(strip4 块1, B 行 64-79) → flush → C 参考对照
        static double tmp[16 * 16];
        double* out = tmp;
        seed(gCB1);
        const double* Ar = gA + diag_strip_off(4) + 1024;
        const double* Br = gCB1 + 64 * NCP;
        sme_start();
        sme_zero_za();
        dd_b0_rem(Ar, Br, 16);
        dd_flush_b0_set(out, 128);   // cs=128B（16 double 紧密行）
        sme_stop();
        long bad = 0;
        for (long r = 0; r < 16; r++) for (long cc = 0; cc < 16; cc++) {
            double ref = 0;
            for (long k = 0; k < 16; k++) ref += Ar[k * 16 + r] * Br[k * NCP + cc];
            double got = out[r * 16 + cc];
            if (ref != got) { if (!bad) printf("UNIT rem first diff (r%ld,c%ld) ref=%.17g got=%.17g\n", r, cc, ref, got); bad++; }
        }
        printf("UNIT rem: %s（bad=%ld/256）\n", bad ? "FAIL" : "PASS", bad);
        const double* Af = gA + diag_strip_off(4);
        const double* Bf = gCB1;
        sme_start();
        sme_zero_za();
        dd_b0_first(Af, Bf, gxpan, (const char*)g_dum, (const char*)g_dum, (const char*)g_dum);
        dd_flush_b0_set(out, 128);
        sme_stop();
        bad = 0;
        for (long r = 0; r < 16; r++) for (long cc = 0; cc < 16; cc++) {
            double ref = 0;
            for (long k = 0; k < 64; k++) ref += Af[k * 16 + r] * Bf[k * NCP + cc];
            if (ref != out[r * 16 + cc]) { if (!bad) printf("UNIT first diff (r%ld,c%ld) ref=%.17g got=%.17g\n", r, cc, ref, out[r * 16 + cc]); bad++; }
        }
        printf("UNIT first: %s（bad=%ld/256）\n", bad ? "FAIL" : "PASS", bad);
        // UNIT2：F; R0 真实对子（K=80 的 bank0 全累加）
        sme_start();
        sme_zero_za();
        dd_b0_first(Af, Bf, gxpan, (const char*)g_dum, (const char*)g_dum, (const char*)g_dum);
        dd_b0_rem(Ar, Br, 16);
        dd_flush_b0_set(out, 128);
        sme_stop();
        bad = 0;
        for (long r = 0; r < 16; r++) for (long cc = 0; cc < 16; cc++) {
            double ref = 0;
            for (long k = 0; k < 64; k++) ref += Af[k * 16 + r] * Bf[k * NCP + cc];
            for (long k = 0; k < 16; k++) ref += Ar[k * 16 + r] * Br[k * NCP + cc];
            if (ref != out[r * 16 + cc]) { if (!bad) printf("UNIT2 diff (r%ld,c%ld) ref=%.17g got=%.17g\n", r, cc, ref, out[r * 16 + cc]); bad++; }
        }
        printf("UNIT2 F+R0: %s（bad=%ld/256）\n", bad ? "FAIL" : "PASS", bad);
        // UNIT3（修正布局：1024B 行距缓冲）：F; R0; ZL → ZL 排水 za0-3 到 cur1024
        double* cur3 = gCB2;                       // 借 gCB2 头部 16KB 当 1024 行距目标
        memset(gCB2, 0, 32 * 1024);
        sme_start();
        sme_zero_za();
        dd_b0_first(Af, Bf, gxpan, (const char*)g_dum, (const char*)g_dum, (const char*)g_dum);
        dd_b0_rem(Ar, Br, 16);
        dd_b4_zlast(Af, Bf + 16, cur3, (const char*)g_dum, (const char*)g_dum, (const char*)g_dum);
        sme_stop();
        bad = 0;
        for (long r = 0; r < 16; r++) for (long cc = 0; cc < 16; cc++) {
            double ref = 0;
            for (long k = 0; k < 64; k++) ref += Af[k * 16 + r] * Bf[k * NCP + cc];
            for (long k = 0; k < 16; k++) ref += Ar[k * 16 + r] * Br[k * NCP + cc];
            double got = cur3[r * NCP + cc];
            if (ref != got) { if (!bad) printf("UNIT3 diff (r%ld,c%ld) ref=%.17g got=%.17g\n", r, cc, ref, got); bad++; }
        }
        printf("UNIT3 F+R0+ZL 排水: %s（bad=%ld/256）\n", bad ? "FAIL" : "PASS", bad);
        // UNIT4：带真实 pending za4-7 的 F 链（zpure 造 pending → F 排水 + 自算 → 双验证）
        double* drn = gCB2 + 4 * 1024;             // 1024 行距排水目标（16KB 区）
        memset(drn, 0, 32 * 1024);
        sme_start();
        sme_zero_za();
        dd_b4_zpure(Af, Bf + 16, gxpan, (const char*)g_dum, (const char*)g_dum, (const char*)g_dum);
        dd_b0_first(Af, Bf, drn, (const char*)g_dum, (const char*)g_dum, (const char*)g_dum);
        dd_flush_b0_set(out, 128);
        sme_stop();
        bad = 0;
        for (long r = 0; r < 16; r++) for (long cc = 0; cc < 16; cc++) {
            double ref = 0;
            for (long k = 0; k < 64; k++) ref += Af[k * 16 + r] * Bf[k * NCP + 16 + cc];
            if (ref != drn[r * NCP + cc]) { if (!bad) printf("UNIT4a diff (r%ld,c%ld) ref=%.17g got=%.17g\n", r, cc, ref, drn[r * NCP + cc]); bad++; }
        }
        printf("UNIT4a F 排水 pending za4-7: %s（bad=%ld/256）\n", bad ? "FAIL" : "PASS", bad);
        bad = 0;
        for (long r = 0; r < 16; r++) for (long cc = 0; cc < 16; cc++) {
            double ref = 0;
            for (long k = 0; k < 64; k++) ref += Af[k * 16 + r] * Bf[k * NCP + cc];
            if (ref != out[r * 16 + cc]) { if (!bad) printf("UNIT4b diff (r%ld,c%ld) ref=%.17g got=%.17g\n", r, cc, ref, out[r * 16 + cc]); bad++; }
        }
        printf("UNIT4b F 自算 za0-3: %s（bad=%ld/256）\n", bad ? "FAIL" : "PASS", bad);
        return 0;
    } }
    // ---- bitwise：全任务 s=0..15，旧 vs（旧 s<=2 + 新 s>=3）——流式区内零库调用，先算后比
    seed(gCB1); seed(gCB2);
    {
        sme_start();
        double* pc1 = NULL; long pcols1 = 0; int pend1 = 0;
        for (int s = 0; s < SS; s++) pend1 = old_strip(s, gCB1, pend1, &pc1, &pcols1, s == SS - 1);
        double* pc2 = NULL; long pcols2 = 0; int pend2 = 0;
        for (int s = 0; s <= 2; s++) pend2 = old_strip(s, gCB2, pend2, &pc2, &pcols2, s == 2);
        double* d4c = gxpan;
        int newmax = g_newmax;
        for (int s = 3; s < SS && s <= newmax; s++) new_strip(s, gCB2, &d4c);
        dd_flush_b4_set(d4c, CS);
        for (int s = (newmax < 3 ? 3 : newmax + 1); s < SS; s++) { int pd = 0; double* pcx = NULL; long pcc = 0; pd = old_strip(s, gCB2, pd, &pcx, &pcc, 1); }
        sme_stop();
    }
    int bitok = memcmp(gCB1, gCB2, 256 * NCP * 8) == 0;
    long diffrow = -1, diffcol = -1; double dv1 = 0, dv2 = 0; long ndiff = 0;
    if (!bitok) for (long r = 0; r < 256; r++) for (long cjj = 0; cjj < NCP; cjj++)
        if (gCB1[r * NCP + cjj] != gCB2[r * NCP + cjj]) {
            if (diffrow < 0) { diffrow = r; diffcol = cjj; dv1 = gCB1[r * NCP + cjj]; dv2 = gCB2[r * NCP + cjj]; }
            ndiff++;
        }
    // ---- 计时：strips 3..15，单缓冲交替（消热度偏置）----
    const int R = 9;
    double olds[R], news[R];
    for (int r = 0; r < R; r++) {
        seed(gCB1);
        sme_start();
        double* pc1 = NULL; long pcols1 = 0; int pend1 = 0;
        for (int s = 0; s <= 2; s++) pend1 = old_strip(s, gCB1, pend1, &pc1, &pcols1, s == 2);
        uint64_t t0 = cyc();
        pend1 = 0;
        for (int s = 3; s < SS; s++) pend1 = old_strip(s, gCB1, pend1, &pc1, &pcols1, s == SS - 1);
        uint64_t t1 = cyc();
        sme_stop();
        olds[r] = (double)(t1 - t0) * c2c;
        seed(gCB1);
        sme_start();
        pc1 = NULL; pcols1 = 0; pend1 = 0;
        for (int s = 0; s <= 2; s++) pend1 = old_strip(s, gCB1, pend1, &pc1, &pcols1, s == 2);
        uint64_t t2 = cyc();
        double* d4c = gxpan;
        for (int s = 3; s < SS; s++) new_strip(s, gCB1, &d4c);
        dd_flush_b4_set(d4c, CS);
        uint64_t t3 = cyc();
        sme_stop();
        news[r] = (double)(t3 - t2) * c2c;
    }
    {   // 诊断轮：分桶计时（不进 W；memset 在区外）
        memset(t_b0, 0, sizeof t_b0); memset(t_b4, 0, sizeof t_b4);
        memset((void*)n_b0, 0, sizeof n_b0); memset((void*)n_b4, 0, sizeof n_b4);
        g_tic = 1;
        sme_start();
        double* d4c = gxpan;
        for (int s = 3; s < SS; s++) new_strip(s, gCB2, &d4c);
        dd_flush_b4_set(d4c, CS);
        sme_stop();
        g_tic = 0;
    }
    sme_stop();
    // 中位
    for (int i = 0; i < R; i++) for (int j = i + 1; j < R; j++) {
        if (olds[j] < olds[i]) { double x = olds[i]; olds[i] = olds[j]; olds[j] = x; }
        if (news[j] < news[i]) { double x = news[i]; news[i] = news[j]; news[j] = x; }
    }
    double om = olds[R / 2], nm = news[R / 2];
    printf("bitwise: %s%s\n", bitok ? "PASS" : "FAIL", bitok ? "" : "（见 diffrow）");
    if (!bitok) {
        printf("  first diff (row %ld, col %ld) old=%.17g new=%.17g；共 %ld 元素异（块行=%ld 块列=%ld）\n",
            diffrow, diffcol, dv1, dv2, ndiff, diffrow / 16, diffcol / 32);
        long r0 = (diffrow / 16) * 16;
        printf("  条 %ld 行 [%ld,%ld) 各 16 列块 diff 数：", diffrow / 16, r0, r0 + 16);
        for (long cb16 = 0; cb16 < NCP; cb16 += 16) {
            long nd = 0;
            for (long r = r0; r < r0 + 16; r++) for (long j = cb16; j < cb16 + 16; j++)
                if (gCB1[r * NCP + j] != gCB2[r * NCP + j]) nd++;
            printf(" %ld", nd);
        }
        printf("\n");
    }
    long ksteps = 0; for (int s = 3; s < SS; s++) ksteps += 4 * 16 * (s + 1);
    printf("== strips 3..15（%ld k 步 ×4 瓦片）==\n", ksteps);
    printf("old（compact+za_dump+chain）: %.0f cyc  %.2f cyc/步\n", om, om / ksteps);
    printf("new（双bank K64 SET 链）    : %.0f cyc  %.2f cyc/步\n", nm, nm / ksteps);
    printf("加权差 W = %+.2f%%（判据：<=-8%% GO / -5~-8 灰区 / 其余弃）\n", (nm / om - 1) * 100);
    const char* bn0[4] = { "b0_first", "b0_pure", "b0_rem", "-" };
    const char* bn4[4] = { "-", "b4_pure", "b4_rem", "b4_drain(zl/zp/last)" };
    printf("== 调用型桶（cyc/调用，b4 vs b0 = L1 重读指纹）==\n");
    for (int i = 0; i < 4; i++) if (n_b0[i]) printf("  %-9s n=%-4lu avg=%.0f\n", bn0[i], (unsigned long)n_b0[i], (double)t_b0[i] / n_b0[i] * c2c);
    for (int i = 0; i < 4; i++) if (n_b4[i]) printf("  %-20s n=%-4lu avg=%.0f\n", bn4[i], (unsigned long)n_b4[i], (double)t_b4[i] / n_b4[i] * c2c);
    return 0;
}
