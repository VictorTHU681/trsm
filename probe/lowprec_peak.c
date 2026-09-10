// 低精度外积吞吐探针：回答"量化/低精度能否比 FP64 FMOPA 更快"这一问题的硬件侧。
// 对每种 SME 外积指令做 8 条/迭代的寄存器内轮转（无访存），报告每核 指令/ns 与 GMAC/s；FP64 8 tile 轮转作基准。
// 编译：gcc -O2 -march=armv9-a+sme+sme-f64f64+sme-i16i64+sve2 -fopenmp probe/lowprec_peak.c -o probe/lowprec_peak
// 运行：OMP_NUM_THREADS=38 OMP_PROC_BIND=close OMP_PLACES=cores numactl -N 1 ./probe/lowprec_peak
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/auxv.h>
#include <omp.h>

#define HWCAP2_SME          (1UL << 23)
#define HWCAP2_SME_I16I64   (1UL << 24)
#define HWCAP2_SME_F64F64   (1UL << 25)
#define HWCAP2_SME_I8I32    (1UL << 26)
#define HWCAP2_SME_F16F32   (1UL << 27)
#define HWCAP2_SME_B16F32   (1UL << 28)
#define HWCAP2_SME_F32F32   (1UL << 29)

#define VCLOB "v0","v1","v2","v3","v4","v5","v6","v7","v8","v9","v10","v11","v12","v13","v14","v15", \
              "v16","v17","v18","v19","v20","v21","v22","v23","v24","v25","v26","v27","v28","v29","v30","v31"
#define PCLOB "p0","p1","p2","p3","p4","p5","p6","p7","p8","p9","p10","p11","p12","p13","p14","p15"

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// 生成一个探针函数：SETUP 为谓词/寄存器初始化，BODY 为 8 条外积指令。返回每秒指令数（G instr/s）。
#define DEF_PROBE(name, SETUP, BODY)                                              \
    static double name(long iters)                                                \
    {                                                                             \
        long n = iters;                                                           \
        double t0 = now();                                                        \
        asm volatile("smstart\n\t" SETUP "zero {za}\n\t"                          \
                     "1:\n\t" BODY                                                \
                     "subs %0, %0, #1\n\t"                                        \
                     "b.ne 1b\n\t"                                                \
                     "smstop\n\t"                                                 \
                     : "+r"(n) :: "memory", "cc", VCLOB, PCLOB);                  \
        double t1 = now();                                                        \
        return (double)iters * 8.0 / (t1 - t0) * 1e-9;                            \
    }

#define OP8_D(op, s)  op " za0.d, p0/m, p1/m, z0." s ", z1." s "\n\t" op " za1.d, p0/m, p1/m, z0." s ", z1." s "\n\t" \
                      op " za2.d, p0/m, p1/m, z0." s ", z1." s "\n\t" op " za3.d, p0/m, p1/m, z0." s ", z1." s "\n\t" \
                      op " za4.d, p0/m, p1/m, z0." s ", z1." s "\n\t" op " za5.d, p0/m, p1/m, z0." s ", z1." s "\n\t" \
                      op " za6.d, p0/m, p1/m, z0." s ", z1." s "\n\t" op " za7.d, p0/m, p1/m, z0." s ", z1." s "\n\t"
#define OP8_D4(op, s) op " za0.d, p0/m, p1/m, z0." s ", z1." s "\n\t" op " za1.d, p0/m, p1/m, z0." s ", z1." s "\n\t" \
                      op " za2.d, p0/m, p1/m, z0." s ", z1." s "\n\t" op " za3.d, p0/m, p1/m, z0." s ", z1." s "\n\t" \
                      op " za0.d, p0/m, p1/m, z0." s ", z1." s "\n\t" op " za1.d, p0/m, p1/m, z0." s ", z1." s "\n\t" \
                      op " za2.d, p0/m, p1/m, z0." s ", z1." s "\n\t" op " za3.d, p0/m, p1/m, z0." s ", z1." s "\n\t"
#define OP8_S(op, s)  op " za0.s, p0/m, p1/m, z0." s ", z1." s "\n\t" op " za1.s, p0/m, p1/m, z0." s ", z1." s "\n\t" \
                      op " za2.s, p0/m, p1/m, z0." s ", z1." s "\n\t" op " za3.s, p0/m, p1/m, z0." s ", z1." s "\n\t" \
                      op " za0.s, p0/m, p1/m, z0." s ", z1." s "\n\t" op " za1.s, p0/m, p1/m, z0." s ", z1." s "\n\t" \
                      op " za2.s, p0/m, p1/m, z0." s ", z1." s "\n\t" op " za3.s, p0/m, p1/m, z0." s ", z1." s "\n\t"
#define OP8_S2(op, s) op " za0.s, p0/m, p1/m, z0." s ", z1." s "\n\t" op " za1.s, p0/m, p1/m, z0." s ", z1." s "\n\t" \
                      op " za0.s, p0/m, p1/m, z0." s ", z1." s "\n\t" op " za1.s, p0/m, p1/m, z0." s ", z1." s "\n\t" \
                      op " za0.s, p0/m, p1/m, z0." s ", z1." s "\n\t" op " za1.s, p0/m, p1/m, z0." s ", z1." s "\n\t" \
                      op " za0.s, p0/m, p1/m, z0." s ", z1." s "\n\t" op " za1.s, p0/m, p1/m, z0." s ", z1." s "\n\t"

DEF_PROBE(p_f64_8t, "ptrue p0.d\n\t ptrue p1.d\n\t fdup z0.d, #1.0\n\t fdup z1.d, #0.5\n\t", OP8_D("fmopa", "d"))
DEF_PROBE(p_f64_4t, "ptrue p0.d\n\t ptrue p1.d\n\t fdup z0.d, #1.0\n\t fdup z1.d, #0.5\n\t", OP8_D4("fmopa", "d"))
DEF_PROBE(p_f32_4t, "ptrue p0.s\n\t ptrue p1.s\n\t fdup z0.s, #1.0\n\t fdup z1.s, #0.5\n\t", OP8_S("fmopa", "s"))
DEF_PROBE(p_f32_2t, "ptrue p0.s\n\t ptrue p1.s\n\t fdup z0.s, #1.0\n\t fdup z1.s, #0.5\n\t", OP8_S2("fmopa", "s"))
DEF_PROBE(p_i8_4t,  "ptrue p0.b\n\t ptrue p1.b\n\t dup z0.b, #3\n\t dup z1.b, #-5\n\t", OP8_S("smopa", "b"))
DEF_PROBE(p_u8_4t,  "ptrue p0.b\n\t ptrue p1.b\n\t dup z0.b, #3\n\t dup z1.b, #5\n\t", OP8_S("umopa", "b"))
DEF_PROBE(p_bf16_4t,"ptrue p0.h\n\t ptrue p1.h\n\t dup z0.h, #0x3f80\n\t dup z1.h, #0x3f00\n\t", OP8_S("bfmopa", "h"))
DEF_PROBE(p_f16w_4t,"ptrue p0.h\n\t ptrue p1.h\n\t dup z0.h, #0x3c00\n\t dup z1.h, #0x3800\n\t", OP8_S("fmopa", "h"))
#ifdef HAVE_I16I64
DEF_PROBE(p_i16_8t, "ptrue p0.h\n\t ptrue p1.h\n\t dup z0.h, #3\n\t dup z1.h, #-5\n\t", OP8_D("smopa", "h"))
#endif

typedef struct { const char* name; double (*fn)(long); double macs_per_instr; unsigned long need; } probe_t;

int main(void)
{
    unsigned long hw2 = getauxval(AT_HWCAP2);
    int nth = omp_get_max_threads();
    printf("HWCAP2: SME=%d I16I64=%d F64F64=%d I8I32=%d F16F32=%d B16F32=%d F32F32=%d  threads=%d\n",
           !!(hw2 & HWCAP2_SME), !!(hw2 & HWCAP2_SME_I16I64), !!(hw2 & HWCAP2_SME_F64F64), !!(hw2 & HWCAP2_SME_I8I32),
           !!(hw2 & HWCAP2_SME_F16F32), !!(hw2 & HWCAP2_SME_B16F32), !!(hw2 & HWCAP2_SME_F32F32), nth);
    // SVL=512：.d tile 8×8，.s tile 16×16；int8 4 路点积、int16 4 路、bf16/fp16 2 路
    probe_t probes[] = {
        { "fmopa.d  8 tile (基准)      ", p_f64_8t, 64,   HWCAP2_SME_F64F64 },
        { "fmopa.d  4 tile             ", p_f64_4t, 64,   HWCAP2_SME_F64F64 },
        { "fmopa.s  4 tile (fp32)      ", p_f32_4t, 256,  HWCAP2_SME_F32F32 },
        { "fmopa.s  2 tile (fp32)      ", p_f32_2t, 256,  HWCAP2_SME_F32F32 },
        { "smopa.s  4 tile (int8→i32) ", p_i8_4t,  1024, HWCAP2_SME_I8I32 },
        { "umopa.s  4 tile (uint8→i32)", p_u8_4t,  1024, HWCAP2_SME_I8I32 },
        { "bfmopa.s 4 tile (bf16→f32) ", p_bf16_4t, 512, HWCAP2_SME_B16F32 },
        { "fmopa.s  4 tile (fp16→f32) ", p_f16w_4t, 512, HWCAP2_SME_F16F32 },
#ifdef HAVE_I16I64
        { "smopa.d  8 tile (int16→i64)", p_i16_8t, 256,  HWCAP2_SME_I16I64 },
#endif
    };
    long iters = 4000000;
    int np = (int)(sizeof(probes) / sizeof(probes[0]));
    printf("%-30s %12s %12s %12s %12s\n", "指令", "单核 instr/ns", "单核 GMAC/s", "38核 TMAC/s", "相对 fp64");
    double base = 0;
    for (int i = 0; i < np; i++) {
        if ((hw2 & probes[i].need) != probes[i].need) { printf("%-30s  硬件不支持\n", probes[i].name); continue; }
        double r1 = probes[i].fn(iters);
        double agg = 0;
#pragma omp parallel num_threads(nth) reduction(+ : agg)
        agg += probes[i].fn(iters);
        double gm1 = r1 * probes[i].macs_per_instr;
        double tm = agg * probes[i].macs_per_instr * 1e-3;
        if (i == 0) base = gm1;
        printf("%-30s %12.3f %12.1f %12.2f %11.2fx\n", probes[i].name, r1, gm1, tm, base > 0 ? gm1 / base : 0.0);
    }
    return 0;
}
