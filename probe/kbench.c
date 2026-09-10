// 单线程内核微基准 v2：include trsm.c 取用内部函数。流式模式只在测量段内进入。
// 编译：gcc -O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp kbench.c -o kbench -lm
// 运行：taskset -c 40 ./kbench [kc] [nc] [mb]
#include "../trsm.c"
#include <math.h>
#include <signal.h>
#include <ucontext.h>

static double to_cycles(uint64_t dt) { return (double)dt / (double)cyc_hz() * 1.55e9; }

// ---------------------------------------------------------------- HZ 原型内核（16×16，半 ZA 交替）
#define HZ_KS(a0, a1, b0, b1, n0, n1, m0, m1, T0, T1, T2, T3)                      \
    "ld1d " n0 ".d, p0/z, [%[xa]]\n\t"                                            \
    "fmopa " T0 ".d, p0/m, p0/m, " a0 ".d, " b0 ".d\n\t"                          \
    "ld1d " n1 ".d, p0/z, [%[xa], #1, mul vl]\n\t"                                \
    "fmopa " T1 ".d, p0/m, p0/m, " a0 ".d, " b1 ".d\n\t"                          \
    "ld1d " m0 ".d, p0/z, [%[xb]]\n\t"                                            \
    "fmopa " T2 ".d, p0/m, p0/m, " a1 ".d, " b0 ".d\n\t"                          \
    "ld1d " m1 ".d, p0/z, [%[xb], #1, mul vl]\n\t"                                \
    "fmopa " T3 ".d, p0/m, p0/m, " a1 ".d, " b1 ".d\n\t"                          \
    "add %[xa], %[xa], #128\n\t"                                                  \
    "add %[xb], %[xb], %[bs]\n\t"
#define HZ_FM(a0, a1, b0, b1, T0, T1, T2, T3)                                     \
    "fmopa " T0 ".d, p0/m, p0/m, " a0 ".d, " b0 ".d\n\t"                          \
    "fmopa " T1 ".d, p0/m, p0/m, " a0 ".d, " b1 ".d\n\t"                          \
    "fmopa " T2 ".d, p0/m, p0/m, " a1 ".d, " b0 ".d\n\t"                          \
    "fmopa " T3 ".d, p0/m, p0/m, " a1 ".d, " b1 ".d\n\t"
#define HZ_E(T0, T1, T2, T3) HZ_KS("z0", "z1", "z4", "z5", "z8", "z9", "z12", "z13", T0, T1, T2, T3)
#define HZ_O(T0, T1, T2, T3) HZ_KS("z8", "z9", "z12", "z13", "z0", "z1", "z4", "z5", T0, T1, T2, T3)
// 转存另一半的一行：D0/D1 为该半的 tile 号立即数。编码要求内存偏移立即数 == 向量选择立即数，
// 因此地址为 xd + D*64：变体 B（tile 0-3）xd 指向行槽起点；变体 A（tile 4-7）xd 指向行槽起点 - 256。
#define HZ_STR(D0, D1)                                                            \
    "str za[w12, " D0 "], [%[xd], #" D0 ", mul vl]\n\t"                           \
    "str za[w12, " D1 "], [%[xd], #" D1 ", mul vl]\n\t"
// 回写一行（两个向量）：前半（装载）/后半（减、存、推进）
#define HZ_EV1                                                                    \
    "ld1d z16.d, p0/z, [%[xs]]\n\t"                                               \
    "ld1d z17.d, p0/z, [%[xs], #1, mul vl]\n\t"                                   \
    "ld1d z18.d, p1/z, [%[xc]]\n\t"                                               \
    "ld1d z19.d, p2/z, [%[xc], #1, mul vl]\n\t"
#define HZ_EV2                                                                    \
    "fsub z18.d, z18.d, z16.d\n\t"                                                \
    "fsub z19.d, z19.d, z17.d\n\t"                                                \
    "st1d z18.d, p1, [%[xc]]\n\t"                                                 \
    "st1d z19.d, p2, [%[xc], #1, mul vl]\n\t"                                     \
    "add %[xs], %[xs], #256\n\t"                                                  \
    "add %[xc], %[xc], %[cs]\n\t"
// 一组 = 4 步（E O E O）：托管 4 条 str（另一半一行 4 个 tile）+ 一行回写；每两组推进 w12（ZA 行）
#define HZ_GROUP(T0, T1, T2, T3, D0, D1, D2, D3)                                  \
    HZ_E(T0, T1, T2, T3) HZ_STR(D0, D1)                                           \
    HZ_O(T0, T1, T2, T3) HZ_EV1                                                   \
    HZ_E(T0, T1, T2, T3) HZ_STR(D2, D3) "add w12, w12, #8\n\t" "add %[xd], %[xd], #256\n\t" \
    HZ_O(T0, T1, T2, T3) HZ_EV2
// 无转存的组（另一半的 32 条 str 在前 8 组已写完）
#define HZ_GROUP_NS(T0, T1, T2, T3)                                               \
    HZ_E(T0, T1, T2, T3)                                                          \
    HZ_O(T0, T1, T2, T3) HZ_EV1                                                   \
    HZ_E(T0, T1, T2, T3)                                                          \
    HZ_O(T0, T1, T2, T3) HZ_EV2
#define HZ_G4(T0, T1, T2, T3, D0, D1, D2, D3)                                     \
    HZ_GROUP(T0, T1, T2, T3, D0, D1, D2, D3) HZ_GROUP(T0, T1, T2, T3, D0, D1, D2, D3) \
    HZ_GROUP(T0, T1, T2, T3, D0, D1, D2, D3) HZ_GROUP(T0, T1, T2, T3, D0, D1, D2, D3)
#define HZ_G8(T0, T1, T2, T3, D0, D1, D2, D3) HZ_G4(T0, T1, T2, T3, D0, D1, D2, D3) HZ_G4(T0, T1, T2, T3, D0, D1, D2, D3)
// 末组：第 4 步只算不装载（K=64 精确）
#define HZ_GROUP_LAST(T0, T1, T2, T3)                                             \
    HZ_E(T0, T1, T2, T3)                                                          \
    HZ_O(T0, T1, T2, T3) HZ_EV1                                                   \
    HZ_E(T0, T1, T2, T3)                                                          \
    HZ_FM("z8", "z9", "z12", "z13", T0, T1, T2, T3) HZ_EV2
#define HZ_G4NS(T0, T1, T2, T3) HZ_GROUP_NS(T0, T1, T2, T3) HZ_GROUP_NS(T0, T1, T2, T3) HZ_GROUP_NS(T0, T1, T2, T3) HZ_GROUP_NS(T0, T1, T2, T3)
#define HZ_G8NS(T0, T1, T2, T3) HZ_G4NS(T0, T1, T2, T3) HZ_GROUP_NS(T0, T1, T2, T3) HZ_GROUP_NS(T0, T1, T2, T3) HZ_GROUP_NS(T0, T1, T2, T3) HZ_GROUP_LAST(T0, T1, T2, T3)
// 变体：TILES = 本半的 4 个 tile；D0..D3 = 另一半的 4 个 tile 号（字符串）。K ≥ 64（前 64 步托管）。
// 转存布局：行 r → 4 个向量 (D0,D1,D2,D3) 连续 256 B；回写读同布局（行 0-7 用向量 0,1；行 8-15 用向量 2,3）
#define DEF_HZ(NAME, T0, T1, T2, T3, D0, D1, D2, D3, XDINIT)                                                                    \
static void NAME(const double* pa, const double* pb, long bs, long K, double* dump, const double* src,        \
                 double* c, long cs, long ecols, int do_dump)                                                  \
{                                                                                                             \
    const double* xa = pa; const double* xb = pb;                                                             \
    char* xd = (char*)dump; const char* xs = (const char*)src; char* xc = (char*)c;                            \
    (void)do_dump; (void)K;   /* 原型：K 必须为 64 */                                                        \
    asm volatile(                                                                                             \
        "whilelo p1.d, xzr, %[ecols]\n\t"                                                                     \
        "subs x9, %[ecols], #8\n\t"                                                                           \
        "csel x9, x9, xzr, gt\n\t"                                                                            \
        "whilelo p2.d, xzr, x9\n\t"                                                                           \
        "ptrue p0.d\n\t"                                                                                      \
        "ld1d z0.d, p0/z, [%[xa]]\n\t"                                                                        \
        "ld1d z1.d, p0/z, [%[xa], #1, mul vl]\n\t"                                                            \
        "ld1d z4.d, p0/z, [%[xb]]\n\t"                                                                        \
        "ld1d z5.d, p0/z, [%[xb], #1, mul vl]\n\t"                                                            \
        "add %[xa], %[xa], #128\n\t"                                                                          \
        "add %[xb], %[xb], %[bs]\n\t"                                                                         \
        "mov w12, #0\n\t"                                                                                     \
        XDINIT \
        HZ_G8(T0, T1, T2, T3, D0, D1, D2, D3)                                                                          \
        "sub %[xs], %[xs], #1920\n\t"                                                                         \
        HZ_G8NS(T0, T1, T2, T3)                                                                               \
        : [xa] "+&r"(xa), [xb] "+&r"(xb), [xd] "+&r"(xd), [xs] "+&r"(xs), [xc] "+&r"(xc)          \
        : [bs] "r"(bs), [cs] "r"(cs), [ecols] "r"(ecols)                                                      \
        : "memory", "cc", "x9", "x12", VCLOB, PCLOB);                                                         \
}
DEF_HZ(hz_kloop_A, "za0", "za1", "za2", "za3", "4", "5", "6", "7", "sub %[xd], %[xd], #256\n\t")
DEF_HZ(hz_kloop_B, "za4", "za5", "za6", "za7", "0", "1", "2", "3", "")
// 注：上面的转存顺序有意为：前 8 组写行 r 的 (D0,D1) 与 (D2,D3)（即每行 4 个向量 256 B），共 8 行 = 2048 B；
// 后 8 组把 xd 退回起点后再写一遍同样 8 行（重复，仅为保持每组结构一致；正确性由第一遍保证，第二遍写相同数据）。
// 独立转存一半（收尾用）：布局同上
static void hz_dump_half0(double* dst)
{
    asm volatile(
        "mov w12, #0\n\t"
        ".rept 8\n\t"
        "str za[w12, 0], [%[d]]\n\t str za[w12, 1], [%[d], #1, mul vl]\n\t str za[w12, 2], [%[d], #2, mul vl]\n\t str za[w12, 3], [%[d], #3, mul vl]\n\t"
        "add %[d], %[d], #256\n\t add w12, w12, #8\n\t"
        ".endr\n\t"
        : [d] "+&r"(dst) : : "memory", "x12");
}
static void hz_dump_half1(double* dst)
{
    asm volatile(
        "mov w12, #0\n\t"
        "sub %[d], %[d], #256\n\t"
        ".rept 8\n\t"
        "str za[w12, 4], [%[d], #4, mul vl]\n\t str za[w12, 5], [%[d], #5, mul vl]\n\t str za[w12, 6], [%[d], #6, mul vl]\n\t str za[w12, 7], [%[d], #7, mul vl]\n\t"
        "add %[d], %[d], #256\n\t add w12, w12, #8\n\t"
        ".endr\n\t"
        : [d] "+&r"(dst) : : "memory", "x12");
}
// 独立回写（收尾用）：C(16×ecols) -= 转存（行 0-7：向量 0,1；行 8-15：向量 2,3）
static void hz_ev_std(const double* src, double* c, long cs, long ecols)
{
    svbool_t p1 = svwhilelt_b64(0L, ecols), p2 = svwhilelt_b64(8L, ecols);
    for (int r = 0; r < 16; r++) {
        const double* v = src + (r < 8 ? 4 * r : 4 * (r - 8) + 2) * 8;
        double* cr = (double*)((char*)c + r * cs);
        svst1_f64(p1, cr, svsub_f64_x(p1, svld1_f64(p1, cr), svld1_f64(p1, v)));
        svst1_f64(p2, cr + 8, svsub_f64_x(p2, svld1_f64(p2, cr + 8), svld1_f64(p2, v + 8)));
    }
}

// 按 K 分派的托管回写 K 循环（kc=64 用专用变体，<64 退化为独立回写 + 普通 K 循环）
static inline void kloop_epi_sub_any(const double* pa, long ase, long aso, const double* pb, long bs, long K, long cols, const pf_t* pf,
                                     const double* scratch, double* c, long cs, long ecols)
{
    if (K >= 65) kloop_epi_sub(pa, ase, aso, pb, bs, K, cols, pf, scratch, c, cs, ecols);
    else if (K == 64) kloop_epi64_sub(pa, ase, aso, pb, bs, cols, pf, scratch, c, cs, ecols);
    else { epi_sub_std(scratch, c, cs, ecols); kloop_plain(pa, ase, aso, pb, bs, K, cols, pf); }
}

// 可行性：K 循环每 2 步插入 2 ld1d + 5 permute(zip/tbl2) + 1 st1d（模拟交错转置的指令密度）
#define KS_TR(a0, a1, b0, b1, b2, b3, n0, n1, m0, m1, m2, m3, ADV)               \
    "ld1d " n0 ".d, p0/z, [%[xa]]\n\t"                                            \
    "fmopa za0.d, p0/m, p0/m, " a0 ".d, " b0 ".d\n\t"                             \
    "ld1d z20.d, p0/z, [%[xt]]\n\t"                                               \
    "fmopa za1.d, p0/m, p0/m, " a0 ".d, " b1 ".d\n\t"                             \
    "ld1d " n1 ".d, p0/z, [%[xa], #1, mul vl]\n\t"                                \
    "zip1 z22.d, z20.d, z21.d\n\t"                                                \
    "fmopa za2.d, p0/m, p0/m, " a0 ".d, " b2 ".d\n\t"                             \
    "tbl z23.d, {z22.d, z23.d}, z28.d\n\t"                                        \
    "fmopa za3.d, p0/m, p0/m, " a0 ".d, " b3 ".d\n\t"                             \
    "ld1d " m0 ".d, p0/z, [%[xb]]\n\t"                                            \
    "zip2 z24.d, z20.d, z21.d\n\t"                                                \
    "fmopa za4.d, p0/m, p0/m, " a1 ".d, " b0 ".d\n\t"                             \
    "ld1d " m1 ".d, p0/z, [%[xb], #1, mul vl]\n\t"                                \
    "fmopa za5.d, p0/m, p0/m, " a1 ".d, " b1 ".d\n\t"                             \
    "ld1d " m2 ".d, p0/z, [%[xb], #2, mul vl]\n\t"                                \
    "tbl z25.d, {z24.d, z25.d}, z29.d\n\t"                                        \
    "fmopa za6.d, p0/m, p0/m, " a1 ".d, " b2 ".d\n\t"                             \
    "ld1d " m3 ".d, p0/z, [%[xb], #3, mul vl]\n\t"                                \
    "fmopa za7.d, p0/m, p0/m, " a1 ".d, " b3 ".d\n\t"                             \
    "st1d z25.d, p0, [%[xt], #1, mul vl]\n\t"                                     \
    "tbl z26.d, {z22.d, z23.d}, z30.d\n\t"                                        \
    "add %[xa], %[xa], " ADV "\n\t"                                               \
    "add %[xb], %[xb], %[bs]\n\t"
static inline void kloop_tr_test(const double* pa, const double* pb, long bs, long K, double* tbuf)
{
    const double* xa = pa; const double* xb = pb; long k = K - 1; double* xt = tbuf;
    asm volatile(
        "ptrue p0.d\n\t"
        "ld1d z0.d, p0/z, [%[xa]]\n\t"
        "ld1d z1.d, p0/z, [%[xa], #1, mul vl]\n\t"
        "ld1d z4.d, p0/z, [%[xb]]\n\t"
        "ld1d z5.d, p0/z, [%[xb], #1, mul vl]\n\t"
        "ld1d z6.d, p0/z, [%[xb], #2, mul vl]\n\t"
        "ld1d z7.d, p0/z, [%[xb], #3, mul vl]\n\t"
        "add %[xa], %[xa], #128\n\t"
        "add %[xb], %[xb], %[bs]\n\t"
        "1:\n\t"
        KS_TR("z0", "z1", "z4", "z5", "z6", "z7", "z8", "z9", "z12", "z13", "z14", "z15", "#128")
        "subs %[k], %[k], #1\n\t"
        "b.eq 5f\n\t"
        KS_TR("z8", "z9", "z12", "z13", "z14", "z15", "z0", "z1", "z4", "z5", "z6", "z7", "#128")
        "subs %[k], %[k], #1\n\t"
        "b.ne 1b\n\t"
        FM_EVEN "b 7f\n\t"
        "5:\n\t" FM_ODD
        "7:\n\t"
        : [xa] "+&r"(xa), [xb] "+&r"(xb), [k] "+&r"(k)
        : [bs] "r"(bs), [xt] "r"(xt)
        : "memory", "cc", VCLOB, PCLOB);
}

static void on_sigill(int sig, siginfo_t* si, void* uc_)
{
    ucontext_t* uc = (ucontext_t*)uc_;
    unsigned long pc = uc->uc_mcontext.pc;
    fprintf(stderr, "SIG%d at pc=%#lx insn=%08x\n", sig, pc, *(unsigned int*)pc);
    _exit(1);
}

int main(int argc, char** argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    struct sigaction sa; memset(&sa, 0, sizeof sa); sa.sa_sigaction = on_sigill; sa.sa_flags = SA_SIGINFO;
    sigaction(SIGILL, &sa, NULL); sigaction(SIGSEGV, &sa, NULL); sigaction(SIGBUS, &sa, NULL); sigaction(SIGFPE, &sa, NULL); sigaction(SIGABRT, &sa, NULL);
    long kc = argc > 1 ? atol(argv[1]) : 128;
    long nc = argc > 2 ? atol(argv[2]) : 128;
    long mb = argc > 3 ? atol(argv[3]) : 256;
    long S = mb / 16, ntile = nc / 32;
    printf("kc=%ld nc=%ld mb=%ld  cpu=%d\n", kc, nc, mb, sched_getcpu());

    double* A = alloc_on(64 * kc * 8, -1);           // 512 B/k 布局
    double* Ad = alloc_on(16 * kc * 8, -1);          // 128 B/k 布局
    double* Xp = alloc_on(kc * nc * 8, -1);
    double* C = alloc_on(mb * nc * 8, -1);
    double* scratch = alloc_on(8192, -1);
    double* dummy = alloc_on(4096, -1);
    memset(dummy, 0, 4096);
    for (long i = 0; i < 64 * kc; i++) A[i] = 1e-3 * (i % 17);
    for (long i = 0; i < 16 * kc; i++) Ad[i] = 1e-3 * (i % 17);
    for (long i = 0; i < kc * nc; i++) Xp[i] = 1e-3 * (i % 13);
    for (long i = 0; i < mb * nc; i++) C[i] = 1.0;
    memset(scratch, 0, 8192);
    pf_t pfd = { (const char*)dummy, 1, -64, (const char*)dummy, (const char*)dummy, 0 };
    uint64_t t0;
    const int reps = 2000;
    double r[16];
    memset(r, 0, sizeof r);
    int only = argc > 4 ? atoi(argv[4]) : 0;   // 第 4 参数 12：只跑 HZ 原型（其他节在 kc<96 时会崩）
    fprintf(stderr, "[kb] only=%d\n", only);
    if (only == 12) goto sec12;

    sme_start();
    for (int w = 0; w < 10; w++) { sme_zero_za(); kloop_plain(Ad, 128, 128, Xp, nc * 8, kc, 32, &pfd); }
    t0 = cyc();
    for (int w = 0; w < reps; w++) { sme_zero_za(); kloop_plain(Ad, 128, 128, Xp, nc * 8, kc, 32, &pfd); }
    r[0] = to_cycles(cyc() - t0) / reps;
    t0 = cyc();
    for (int w = 0; w < reps; w++) { sme_zero_za(); kloop_plain(A, 512, 512, Xp, nc * 8, kc, 32, &pfd); }
    r[1] = to_cycles(cyc() - t0) / reps;
    t0 = cyc();
    for (int w = 0; w < reps; w++) { sme_zero_za(); kloop_plain(Ad, 128, 128, Xp, nc * 8, kc, 32, &pfd); za_dump(scratch); epi_sub_std(scratch, C, nc * 8, 32); }
    r[2] = to_cycles(cyc() - t0) / reps;
    if (kc >= 65) {
        t0 = cyc();
        for (int w = 0; w < reps; w++) { sme_zero_za(); kloop_epi_sub(Ad, 128, 128, Xp, nc * 8, kc, 32, &pfd, scratch, C, nc * 8, 32); za_dump(scratch); }
        r[3] = to_cycles(cyc() - t0) / reps;
        t0 = cyc();
        for (int w = 0; w < reps; w++) { sme_zero_za(); kloop_epi_set(Ad, 128, 128, Xp, nc * 8, kc, 32, &pfd, scratch, C, nc * 8, 32); za_dump(scratch); }
        r[4] = to_cycles(cyc() - t0) / reps;
    } else if (kc == 64) {
        t0 = cyc();
        for (int w = 0; w < reps; w++) { sme_zero_za(); kloop_epi64_sub(Ad, 128, 128, Xp, nc * 8, 32, &pfd, scratch, C, nc * 8, 32); za_dump(scratch); }
        r[3] = to_cycles(cyc() - t0) / reps;
        t0 = cyc();
        for (int w = 0; w < reps; w++) { sme_zero_za(); kloop_epi64_set(Ad, 128, 128, Xp, nc * 8, 32, &pfd, scratch, C, nc * 8, 32); za_dump(scratch); }
        r[4] = to_cycles(cyc() - t0) / reps;
    } else { r[3] = r[4] = 0; }
    sme_stop();
    double* tb2 = alloc_on(8192, -1); memset(tb2, 0, 8192);   // 注意：系统调用会退出流式模式，必须在 smstart 之前分配
    double* Anext = alloc_on(16 * kc * 8, -1); memset(Anext, 0, 16 * kc * 8);
    sme_start();
    double rtr = 0, rtr2 = 0, rtr3 = 0;
    if (kc >= 96 && (kc & 31) == 0) {   // kloop_tr 要求 K ≥ 96 且为 32 的倍数
    t0 = cyc();
    for (int w = 0; w < reps; w++) { sme_zero_za(); kloop_tr_test(Ad, Xp, nc * 8, kc, tb2); }
    rtr = to_cycles(cyc() - t0) / reps;
    // 真正的交错转置 K 循环（源 Ad 当作 16 行 L，行距 kc）：无回写 / 带回写
    t0 = cyc();
    for (int w = 0; w < reps; w++) { sme_zero_za(); kloop_tr_noepi(Ad, Xp, nc * 8, kc, &pfd, Ad, kc * 8, Anext); }
    rtr2 = to_cycles(cyc() - t0) / reps;
    t0 = cyc();
    for (int w = 0; w < reps; w++) { sme_zero_za(); kloop_tr_epi(Ad, Xp, nc * 8, kc, &pfd, scratch, C, nc * 8, 32, Ad, kc * 8, Anext); za_dump(scratch); }
    rtr3 = to_cycles(cyc() - t0) / reps;
    }
    sme_stop();
    // 转置正确性：Anext[k*16 + r] 应等于 Ad[r*kc + k]（源当作 16 行 × kc 列，行距 kc），只覆盖前 kc/4 列… 一个 K 循环转置 K/16 块 = kc/4 列
    { int badt = 0; for (long k = 0; k < kc / 4; k++) for (int rr = 0; rr < 16; rr++) if (Anext[k * 16 + rr] != Ad[rr * kc + k]) badt++;
      printf("1x. kloop_tr 转置正确性（前 %ld 列）: %s\n", kc / 4, badt ? "WRONG" : "ok"); }
    printf("1u. kloop_tr（无回写）             : %8.0f 周期 = %.2f 周期/步\n", rtr2, rtr2 / kc);
    printf("1v. kloop_tr（带回写）+dump         : %8.0f 周期 = %.2f 周期/步\n", rtr3, rtr3 / kc);
    printf("1t. kloop + 每步 1 ld + 2.5 perm + 0.5 st: %8.0f 周期 = %.2f 周期/步\n", rtr, rtr / kc);
    printf("1. kloop_plain A 128B/k          : %8.0f 周期 = %.2f 周期/步\n", r[0], r[0] / kc);
    printf("1b. kloop_plain A 512B/k         : %8.0f 周期 = %.2f 周期/步\n", r[1], r[1] / kc);
    printf("2. kloop + dump + epi_sub_std    : %8.0f 周期 → 额外 %.0f\n", r[2], r[2] - kc * 8.0);
    printf("2h. kloop_epi_sub(宿主回写)+dump  : %8.0f 周期 → 额外 %.0f\n", r[3], r[3] - kc * 8.0);
    printf("2s. kloop_epi_set(宿主回写)+dump  : %8.0f 周期 → 额外 %.0f\n", r[4], r[4] - kc * 8.0);

    // 3. 打包：各变体（源驻留）+ 正确性
    long lda = 17024 + 64;
    double* Lsrc = alloc_on(16 * lda * 8 + kc * 8, -1);
    for (long i = 0; i < 16 * lda; i++) Lsrc[i] = 1e-3 * (i % 7) + i;
    double r5v[4];
    int bad4 = 0, bad2 = 0, badz = 0;
    sme_start();
    for (int w = 0; w < 10; w++) pack_strip_ld4(Lsrc, lda, 0, 0, kc, A);
    t0 = cyc();
    for (int w = 0; w < reps; w++) pack_strip_ld4(Lsrc, lda, 0, 0, kc, A);
    r5v[0] = to_cycles(cyc() - t0) / reps;
    sme_stop();
    for (long k = 0; k < kc; k++) for (int rr = 0; rr < 16; rr++) if (A[k * 16 + rr] != Lsrc[rr * lda + k]) bad4++;
    sme_start();
    t0 = cyc();
    for (int w = 0; w < reps; w++) pack_strip_ld2(Lsrc, lda, 0, 0, kc, A);
    r5v[1] = to_cycles(cyc() - t0) / reps;
    sme_stop();
    for (long k = 0; k < kc; k++) for (int rr = 0; rr < 16; rr++) { long off = (k & 1) ? 32 * k - 16 : 32 * k; if (A[off + rr] != Lsrc[rr * lda + k]) bad2++; }
    sme_start();
    t0 = cyc();
    for (int w = 0; w < reps; w++) pack_strip_za(Lsrc, lda, 0, 0, kc, A);
    r5v[2] = to_cycles(cyc() - t0) / reps;
    sme_stop();
    for (long k = 0; k < kc; k++) for (int rr = 0; rr < 16; rr++) if (A[k * 64 + rr] != Lsrc[rr * lda + k]) badz++;
    sme_start();
    t0 = cyc();
    for (int w = 0; w < reps; w++) pack_strip(Lsrc, lda, 1 << 30, 0, 0, kc, Ad);
    r5v[3] = to_cycles(cyc() - t0) / reps;
    // K 循环用 ld2 布局（步长 128/384）与 ld4 布局
    pack_strip_ld2(Lsrc, lda, 0, 0, kc, A);
    t0 = cyc();
    for (int w = 0; w < reps; w++) { sme_zero_za(); kloop_plain(A, 128, 384, Xp, nc * 8, kc, 32, &pfd); }
    double rk2 = to_cycles(cyc() - t0) / reps;
    sme_stop();
    printf("3a. pack_strip_ld4（稠密 128B/k）   : %8.0f 周期 = %.2f 周期/元素  %s\n", r5v[0], r5v[0] / (16 * kc), bad4 ? "WRONG" : "ok");
    printf("3b. pack_strip_ld2（交替 128/384）   : %8.0f 周期 = %.2f 周期/元素  %s\n", r5v[1], r5v[1] / (16 * kc), bad2 ? "WRONG" : "ok");
    printf("3c. pack_strip_za（512B/k）          : %8.0f 周期 = %.2f 周期/元素  %s\n", r5v[2], r5v[2] / (16 * kc), badz ? "WRONG" : "ok");
    printf("3d. pack_strip（SVE zip/tbl2）       : %8.0f 周期 = %.2f 周期/元素\n", r5v[3], r5v[3] / (16 * kc));
    printf("3k. kloop A=ld2 布局                 : %8.0f 周期 = %.2f 周期/步\n", rk2, rk2 / kc);
    sme_start(); pack_strip_ld4(Lsrc, lda, 0, 0, kc, A); sme_stop();   // 后续段用稠密布局
    // 4. 整段（全驻留）：宿主回写
    double ideal = (double)S * ntile * kc * 8;
    sme_start();
    t0 = cyc();
    for (int w = 0; w < 20; w++) {
        int pend = 0; double* pc = C;
        for (long rr = 0; rr < S; rr++)
            for (long t = 0; t < ntile; t++) {
                sme_zero_za();
                if (pend) kloop_epi_sub_any(A, 128, 128, Xp + 32 * t, nc * 8, kc, 32, &pfd, scratch, pc, nc * 8, 32);
                else kloop_plain(A, 128, 128, Xp + 32 * t, nc * 8, kc, 32, &pfd);
                za_dump(scratch);
                pend = 1; pc = C + 16 * rr * nc + 32 * t;
            }
        epi_sub_std(scratch, pc, nc * 8, 32);
    }
    r[7] = to_cycles(cyc() - t0) / 20;
    // 4b：整段 + 每条 FMOPA 打包（源驻留）
    t0 = cyc();
    for (int w = 0; w < 20; w++) {
        int pend = 0; double* pc = C;
        for (long rr = 0; rr < S; rr++) {
            pack_strip_ld4(Lsrc, lda, 0, 0, kc, A);
            for (long t = 0; t < ntile; t++) {
                sme_zero_za();
                if (pend) kloop_epi_sub_any(A, 128, 128, Xp + 32 * t, nc * 8, kc, 32, &pfd, scratch, pc, nc * 8, 32);
                else kloop_plain(A, 128, 128, Xp + 32 * t, nc * 8, kc, 32, &pfd);
                za_dump(scratch);
                pend = 1; pc = C + 16 * rr * nc + 32 * t;
            }
        }
        epi_sub_std(scratch, pc, nc * 8, 32);
    }
    r[8] = to_cycles(cyc() - t0) / 20;
    sme_stop();
    printf("4. 整段 %ld 条×%ld 块 宿主回写（全驻留）: %8.0f 周期，理想 %.0f，效率 %.1f%%\n", S, ntile, r[7], ideal, ideal / r[7] * 100);
    printf("4b. 同上 + 每条 ld4 打包             : %8.0f 周期，效率 %.1f%%\n", r[8], ideal / r[8] * 100);

    // 5. 流式：X 面板拷贝 + L JIT（源为大数组），带 K 循环内预取
    long m = 17024, ldb = 512 + 64;
    double* Xbig = alloc_on((size_t)m * ldb * 8, -1);
    double* Lbig = alloc_on((size_t)m * lda * 8, -1);
    memset(Xbig, 0, (size_t)m * ldb * 8);
    memset(Lbig, 0, (size_t)m * lda * 8);
    double* Xpan = alloc_on(kc * nc * 8, -1);
    int nchunk = (int)(m / kc) - 2;
    long xlines = ((nc * 8 + 63) / 64 + 1) & ~1L;
    for (int variant = 0; variant < 3; variant++) {
        // variant 0：无预取；1：预取 L 下一条 + X 下一段；2：仅预取 X 下一段
        sme_start();
        t0 = cyc();
        for (int ch = 0; ch < nchunk; ch++) {
            long k0 = (long)ch * kc, k0n = k0 + kc;
            copy_panel(Xbig + k0 * ldb, ldb, kc, nc, nc, Xpan);
            long rows_per_tile = kc / xlines, xrow_next = 0;
            int pend = 0; double* pc = C;
            for (long rr = 0; rr < S; rr++) {
                long row0 = 8192 + 16 * rr;
                pack_strip_ld4(Lbig, lda, row0, k0, kc, A);
                for (long t = 0; t < ntile; t++) {
                    pf_t pf = pfd;
                    if (variant == 1 && t < 2) {
                        long nr = (rr + 1 < S) ? row0 + 16 : 8192, nk = (rr + 1 < S) ? k0 : k0n;
                        pf.p = (const char*)(Lbig + (nr + 8 * t) * lda + nk);
                        pf.pfn = ((kc * 8 + 63) / 64 + 1) & ~1L;
                        pf.pfj = lda * 8 - pf.pfn * 64;
                    } else if (variant >= 1 && xrow_next < kc) {
                        pf.p = (const char*)(Xbig + (k0n + xrow_next) * ldb);
                        pf.pfn = xlines; pf.pfj = ldb * 8 - xlines * 64;
                        xrow_next += rows_per_tile;
                    }
                    sme_zero_za();
                    if (pend) kloop_epi_sub_any(A, 128, 128, Xpan + 32 * t, nc * 8, kc, 32, &pf, scratch, pc, nc * 8, 32);
                    else kloop_plain(A, 128, 128, Xpan + 32 * t, nc * 8, kc, 32, &pf);
                    za_dump(scratch);
                    pend = 1; pc = C + 16 * rr * nc + 32 * t;
                }
            }
            epi_sub_std(scratch, pc, nc * 8, 32);
        }
        r[9 + variant] = to_cycles(cyc() - t0) / nchunk;
        sme_stop();
    }
    // 5d：仅 ld4 打包，源冷（流式读 L 行），无预取
    sme_start();
    t0 = cyc();
    for (int ch = 0; ch < nchunk; ch++) {
        long k0 = (long)ch * kc;
        for (long rr = 0; rr < S; rr++) pack_strip_ld4(Lbig, lda, 8192 + 16 * rr, k0, kc, A);
    }
    r[12] = to_cycles(cyc() - t0) / nchunk;
    // 5e：仅 X 面板拷贝，冷
    t0 = cyc();
    for (int ch = 0; ch < nchunk; ch++) copy_panel(Xbig + (long)ch * kc * ldb, ldb, kc, nc, nc, Xpan);
    r[13] = to_cycles(cyc() - t0) / nchunk;
    sme_stop();
    // 6. 预取有效性：对下一段 X 面板（kc 行 × 1 KB，行距 ldb）先用不同方式预取，再计时 copy_panel
    {
        double rc[6];
        int nrep = 40;
        for (int v = 0; v < 6; v++) {
            double acc = 0;
            for (int rep = 0; rep < nrep; rep++) {
                long ch = 200 + rep * 3 + v * 200;                    // 每次换一段冷数据
                const double* Xk = Xbig + (long)ch * kc * ldb;
                sme_start();
                if (v == 1) {   // K 循环走线器 pldl1keep（8 行/微块，kc/8 个微块）
                    for (long q = 0; q < kc / 8; q++) {
                        pf_t pf = { (const char*)(Xk + q * 8 * ldb), xlines, ldb * 8 - xlines * 64, (const char*)dummy, (const char*)dummy, 0 };
                        sme_zero_za(); kloop_plain(A, 128, 128, Xp, nc * 8, kc, 32, &pf);
                    }
                } else if (v == 2) {   // 突发 prfm pldl1keep
                    for (long k = 0; k < kc; k++) for (long j = 0; j < nc; j += 8) asm volatile("prfm pldl1keep, [%0]" ::"r"(Xk + k * ldb + j));
                } else if (v == 3) {   // 突发 prfm pldl2keep
                    for (long k = 0; k < kc; k++) for (long j = 0; j < nc; j += 8) asm volatile("prfm pldl2keep, [%0]" ::"r"(Xk + k * ldb + j));
                } else if (v == 4) {   // 真实装载触碰（每行 16 条 ld1d 到哑寄存器）
                    for (long k = 0; k < kc; k++) for (long j = 0; j < nc; j += 8) asm volatile("ld1d z31.d, p0/z, [%0]" ::"r"(Xk + k * ldb + j) : "v31");
                } else if (v == 5) {   // K 循环走线器，但每步 2 行连续（把面板当成连续 128 KB：pfn 极大，pfj 0）—— 检验连续目标
                    pf_t pf = { (const char*)Xpan, 1L << 40, 0, (const char*)dummy, (const char*)dummy, 0 };
                    for (long q = 0; q < kc / 8; q++) { sme_zero_za(); kloop_plain(A, 128, 128, Xp, nc * 8, kc, 32, &pf); pf.p += 128 * 64; }
                }
                uint64_t tc0 = cyc();
                copy_panel(v == 5 ? Xpan : Xk, v == 5 ? nc : ldb, kc, nc, nc, Xpan + (v == 5 ? 0 : 0));
                acc += to_cycles(cyc() - tc0);
                sme_stop();
            }
            rc[v] = acc / nrep;
        }
        printf("6. copy_panel %ld KB：冷 %.0f | 走线器 pldl1keep 后 %.0f | 突发 l1keep 后 %.0f | 突发 l2keep 后 %.0f | ld1d 触碰后 %.0f | (连续源热) %.0f 周期\n",
               kc * nc * 8 / 1024, rc[0], rc[1], rc[2], rc[3], rc[4], rc[5]);
    }
    // 7. 复现 run_task 的 GEMM 段循环（Xp 连续面板 + JIT 交错转置 + 双走线器），单线程流式
    {
        long chunks_total = m / kc;             // Xp：chunks × 1 面板（nc 列），连续
        double* Xpp = alloc_on((size_t)chunks_total * kc * nc * 8, -1);
        memset(Xpp, 0, (size_t)chunks_total * kc * nc * 8);
        double* Ab0 = alloc_on(16 * kc * 8, -1), *Ab1 = alloc_on(16 * kc * 8, -1);
        memset(Ab0, 0, 16 * kc * 8); memset(Ab1, 0, 16 * kc * 8);
        double* Abf[2] = { Ab0, Ab1 };
        long row_base = 8192;
        int nch = 40;
        double r7[8];
        for (int variant = 0; variant < 8; variant++) {
            // 0：无预取；1：L 走线器（2 条前）；2：L + X 走线器；3：L 行驻留（源固定 16 行）+ 无预取；4：全部驻留（L 固定 + Xp 固定面板）；5: L 走线器 + X 面板固定；6: X 面板固定 + L 流式无走线器；7: L 走线器 + 任务起点触碰第 1、2 条
            sme_start();
            t0 = cyc();
            long nstrips = nch * S;
            int hosted = 0;
            for (long q = 0; q < nstrips; q++) {
                long ch = q / S; int rr = (int)(q % S);
                long k0 = ch * kc;
                long row0 = row_base + 16 * rr;
                if (variant == 3 || variant == 4) row0 = row_base;                       // L 源固定为同一条
                const double* Xk = Xpp + (size_t)((variant == 4 || variant == 5 || variant == 6) ? 0 : ch) * kc * nc;
                const double* Xkn = Xpp + (size_t)((variant == 4 || variant == 5 || variant == 6) ? 0 : ch + 1) * kc * nc;
                double* Acur = Abf[q & 1];
                if (variant == 7 && q == 0) {   // 触碰第 1、2 条的 L 行（真实装载）
                    for (long qq = 1; qq <= 2; qq++) {
                        long tr0 = row_base + 16 * (qq % S), tk0 = (qq / S) * kc;
                        for (int rr2 = 0; rr2 < 16; rr2++) for (long b = 0; b < kc * 8; b += 64)
                            asm volatile("ld1d z11.d, p0/z, [%0]" ::"r"((const char*)(Lbig + (tr0 + rr2) * lda + tk0) + b) : "v11");
                    }
                }
                if (!hosted) pack_strip_ld4(Lbig, lda, row0, k0, kc, Acur);
                long qn = q + 1; long nrow0 = row_base + 16 * (qn % S); long nk0 = (qn / S) * kc;
                if (variant == 3 || variant == 4) nrow0 = row_base;
                int host_now = (qn < nstrips);
                hosted = host_now;
                long pfq = q + 2;
                int pend = 0; double* pc = C;
                for (long t = 0; t < ntile; t++) {
                    pf_t pf = pfd;
                    if ((variant == 1 || variant == 2 || variant == 5 || variant == 7) && pfq < nstrips) {
                        long pr0 = row_base + 16 * (pfq % S), pk0 = (pfq / S) * kc;
                        long lines_per_row = kc * 8 / 64, rows_per_tile = (kc / 2) / lines_per_row;
                        long rr0 = t * rows_per_tile;
                        if (rr0 < 16) { pf.p = (const char*)(Lbig + (pr0 + rr0) * lda + pk0); pf.pfn = lines_per_row; pf.pfj = lda * 8 - lines_per_row * 64; }
                    }
                    if (variant == 2) {
                        long slots_needed = (kc * nc * 8 + kc * 32 - 1) / (kc * 32), strips_needed = (slots_needed + ntile - 1) / ntile;
                        long xstart = S - strips_needed; if (xstart < 0) xstart = 0;
                        if (rr >= xstart) { long slot = (rr - xstart) * ntile + t; if (slot < slots_needed) pf.p2 = (const char*)Xkn + slot * (kc / 2) * 64; }
                    }
                    if (variant != 0) { pf.pc = (const char*)(C + 16 * rr * nc + 32 * t); pf.pcs = nc * 8; }   // C 行触碰（变体 0 除外）
                    sme_zero_za();
                    long kb0 = t * (kc / 4);
                    if (host_now) {
                        if (pend) kloop_tr_epi(Acur, Xk + 32 * t, nc * 8, kc, &pf, scratch, pc, nc * 8, 32, Lbig + nrow0 * lda + nk0 + kb0, lda * 8, Abf[(q + 1) & 1] + kb0 * 16);
                        else kloop_tr_noepi(Acur, Xk + 32 * t, nc * 8, kc, &pf, Lbig + nrow0 * lda + nk0 + kb0, lda * 8, Abf[(q + 1) & 1] + kb0 * 16);
                    } else if (pend) kloop_epi_sub_any(Acur, 128, 128, Xk + 32 * t, nc * 8, kc, 32, &pf, scratch, pc, nc * 8, 32);
                    else kloop_plain(Acur, 128, 128, Xk + 32 * t, nc * 8, kc, 32, &pf);
                    za_dump(scratch);
                    pend = 1; pc = C + 16 * rr * nc + 32 * t;
                }
                epi_sub_std(scratch, pc, nc * 8, 32);
            }
            r7[variant] = to_cycles(cyc() - t0) / nch;
            sme_stop();
        }
        double id = ideal;
        printf("7. run_task 复现（每段周期，理想 %.0f，TR_LOOKAHEAD=%d）：无预取 %.0f (%.0f%%) | L 走线器 %.0f (%.0f%%) | L+X 走线器 %.0f (%.0f%%) | L 驻留 %.0f (%.0f%%) | 全驻留 %.0f (%.0f%%) | L走线器+X驻留 %.0f (%.0f%%) | X驻留+L流式无走线器 %.0f (%.0f%%) | L走线器+起点触碰 %.0f (%.0f%%)\n",
               id, TR_LOOKAHEAD, r7[0], id / r7[0] * 100, r7[1], id / r7[1] * 100, r7[2], id / r7[2] * 100, r7[3], id / r7[3] * 100, r7[4], id / r7[4] * 100, r7[5], id / r7[5] * 100, r7[6], id / r7[6] * 100, r7[7], id / r7[7] * 100);
    }
    // 8. L 条带（16 行 × kc，行距 lda）的预取有效性：走线器/突发 后 pack_strip_ld4 计时；大数组 Lbig vs 小数组 Lsm（16 MB，确保大页）
    {
        FILE* f = fopen("/proc/self/smaps_rollup", "r"); char line[256];
        if (f) { while (fgets(line, sizeof line, f)) if (strstr(line, "AnonHugePages")) printf("   %s", line); fclose(f); }
        long lda_s = 2048 + 64;                      // 小数组：行距 16.9 KB
        double* Lsm = alloc_on((size_t)1024 * lda_s * 8, -1);
        memset(Lsm, 0, (size_t)1024 * lda_s * 8);
        double r8[8];
        const int nrep = 30;
        for (int v = 0; v < 8; v++) {
            double acc = 0;
            for (int rep = 0; rep < nrep; rep++) {
                int big = v < 4;
                const double* Lx = big ? Lbig : Lsm;
                long ldx = big ? lda : lda_s;
                long row0 = big ? 1024 + rep * 48 + (v % 4) * 3000 : (rep * 32 + (v % 4) * 256) % 900;
                long k0 = big ? 4096 + (v % 4) * kc : 0;
                sme_start();
                int mode = v % 4;   // 0 冷；1 走线器（4 个 K 循环各 4 行）；2 突发 l1keep；3 热（先打包一遍）
                if (mode == 1) {
                    for (int t = 0; t < 4; t++) {
                        pf_t pf = { (const char*)(Lx + (row0 + 4 * t) * ldx + k0), kc * 8 / 64, ldx * 8 - kc * 8, (const char*)dummy, (const char*)dummy, 0 };
                        sme_zero_za(); kloop_plain(Ad, 128, 128, Xp, nc * 8, kc, 32, &pf);
                    }
                } else if (mode == 2) {
                    for (int rr = 0; rr < 16; rr++) for (long b = 0; b < kc * 8; b += 64) asm volatile("prfm pldl1keep, [%0]" ::"r"((const char*)(Lx + (row0 + rr) * ldx + k0) + b));
                } else if (mode == 3) {
                    pack_strip_ld4(Lx, ldx, row0, k0, kc, A);
                }
                uint64_t tp = cyc();
                pack_strip_ld4(Lx, ldx, row0, k0, kc, A);
                acc += to_cycles(cyc() - tp);
                sme_stop();
            }
            r8[v] = acc / nrep;
        }
        printf("8. pack_strip_ld4 16×%ld：Lbig(行距 %ld KB) 冷 %.0f | 走线器后 %.0f | 突发后 %.0f | 热 %.0f || Lsm(行距 %ld KB) 冷 %.0f | 走线器后 %.0f | 突发后 %.0f | 热 %.0f\n",
               kc, lda * 8 / 1024, r8[0], r8[1], r8[2], r8[3], lda_s * 8 / 1024, r8[4], r8[5], r8[6], r8[7]);
        // 8b. 预取 K 循环同时带并发冷 B 流（B 面板从冷区取）时的有效性；以及用 kloop_tr 本身（转置源冷）做宿主
        {
            double r8b[5];
            for (int v = 0; v < 5; v++) {
                double acc = 0;
                for (int rep = 0; rep < nrep; rep++) {
                    long row0 = 1024 + rep * 48 + 400 * v, k0 = 4096;
                    long chb = 300 + rep * 4 + 40 * v;                    // 冷 B 面板
                    const double* Bcold = Xbig + (size_t)chb * kc * ldb;
                    long rowT = 9000 + rep * 48 + 400 * v;                // 冷转置源
                    sme_start();
                    for (int t = 0; t < 4; t++) {
                        pf_t pf = pfd;
                        if (v != 3) { pf.p = (const char*)(Lbig + (row0 + 4 * t) * lda + k0); pf.pfn = kc * 8 / 64; pf.pfj = lda * 8 - kc * 8; }
                        sme_zero_za();
                        if (v == 0) kloop_plain(Ad, 128, 128, Bcold, ldb * 8, kc, 32, &pf);                  // 预取 + 冷 B 流
                        else if (v == 1) kloop_tr_noepi(Ad, Xp, nc * 8, kc, &pf, Lbig + (rowT + 0) * lda + k0 + t * (kc / 4), lda * 8, A);   // 预取 + 冷转置源
                        else if (v == 2) kloop_tr_noepi(Ad, Bcold, ldb * 8, kc, &pf, Lbig + (rowT + 0) * lda + k0 + t * (kc / 4), lda * 8, A); // 预取 + 冷 B + 冷转置
                        else if (v == 3) kloop_plain(Ad, 128, 128, Xp, nc * 8, kc, 32, &pf);                  // 无预取（对照，冷）
                        else kloop_tr_noepi(Ad, Xp, nc * 8, kc, &pf, Ad, kc * 8, A);                 // 预取 + 热转置源
                    }
                    uint64_t tp = cyc();
                    pack_strip_ld4(Lbig, lda, row0, k0, kc, A);
                    acc += to_cycles(cyc() - tp);
                    sme_stop();
                }
                r8b[v] = acc / nrep;
            }
            printf("8b. 走线器预取后 pack：宿主带冷 B 流 %.0f | 宿主带冷转置源 %.0f | 冷 B + 冷转置 %.0f | 无预取对照 %.0f | 宿主带热转置源 %.0f\n", r8b[0], r8b[1], r8b[2], r8b[3], r8b[4]);
        }
    }
    // 9. 预打包路径复现（case2 结构）：A 条连续（预打包，冷流），Xp 面板连续（冷流），cbuf 驻留；走线器变体
    {
        long chunks_total = m / kc;
        double* Xpp = alloc_on((size_t)chunks_total * kc * nc * 8, -1);
        memset(Xpp, 0, (size_t)chunks_total * kc * nc * 8);
        size_t pp_doubles = (size_t)chunks_total * S * 16 * kc;          // 每段 S 条 × 16×kc
        double* PP = alloc_on(pp_doubles * 8, -1);
        memset(PP, 0, pp_doubles * 8);
        int nch = 40;
        double r9[12];
        for (int variant = 0; variant < 12; variant++) {
            // 0 无预取；1 A 走线器；2 A+X 走线器；3 X 走线器；4 A 驻留（固定条）+X 流；5 全驻留；6 A 走线器 + X 驻留；7 A 走线器提前 1 条；8 A+X 走线器 + 起点预热
            // 9 无预取 + C 微块固定（cbuf 足迹 4 KB）；10 A 走线器 + C 固定；11 A+X 走线器 + C 固定；变体 1-3,6-11 均带 C 行 pldl2keep 触碰
            int cfix = variant >= 9;
            sme_start();
            if (variant == 8) {   // 预热：首段面板 + 前 2 条 A（真实装载，阻塞式）
                for (long b = 0; b < kc * nc * 8; b += 64) asm volatile("ld1d z11.d, p0/z, [%0]" ::"r"((const char*)Xpp + b) : "v11");
                for (long b = 0; b < 2 * 16 * kc * 8; b += 64) asm volatile("ld1d z11.d, p0/z, [%0]" ::"r"((const char*)PP + b) : "v11");
            }
            t0 = cyc();
            for (int ch = 0; ch < nch; ch++) {
                const double* Xk = Xpp + (size_t)((variant == 5 || variant == 6) ? 0 : ch) * kc * nc;
                const double* Xkn = Xpp + (size_t)((variant == 5 || variant == 6) ? 0 : ch + 1) * kc * nc;
                int pend = 0; double* pc = C;
                for (int rr = 0; rr < S; rr++) {
                    const double* Acur = PP + ((size_t)((variant == 4 || variant == 5) ? 0 : ch) * S + ((variant == 4 || variant == 5) ? 0 : rr)) * 16 * kc;
                    long qn2 = (long)ch * S + rr + (variant == 7 ? 1 : 2);
                    const double* An = PP + (size_t)qn2 * 16 * kc;
                    for (long t = 0; t < ntile; t++) {
                        pf_t pf = pfd;
                        if (variant == 1 || variant == 2 || variant == 6 || variant == 7 || variant == 8 || variant == 10 || variant == 11) { pf.p = (const char*)An + (size_t)t * (kc / 2) * 64; pf.pfn = 1L << 40; pf.pfj = 0; }
                        if (variant == 2 || variant == 3 || variant == 8 || variant == 11) {
                            long slots_needed = (kc * nc * 8 + kc * 32 - 1) / (kc * 32), strips_needed = (slots_needed + ntile - 1) / ntile;
                            long xstart = S - strips_needed; if (xstart < 0) xstart = 0;
                            if (rr >= xstart) { long slot = (rr - xstart) * ntile + t; if (slot < slots_needed) pf.p2 = (const char*)Xkn + slot * (kc / 2) * 64; }
                        }
                        if (variant != 0 && variant != 5 && variant != 4) { pf.pc = (const char*)(cfix ? C : C + 16 * rr * nc + 32 * t); pf.pcs = nc * 8; }   // C 行 pldl2keep 触碰
                        sme_zero_za();
                        if (pend) kloop_epi_sub_any(Acur, 128, 128, Xk + 32 * t, nc * 8, kc, 32, &pf, scratch, pc, nc * 8, 32);
                        else kloop_plain(Acur, 128, 128, Xk + 32 * t, nc * 8, kc, 32, &pf);
                        za_dump(scratch);
                        pend = 1; pc = cfix ? C : C + 16 * rr * nc + 32 * t;
                    }
                }
                epi_sub_std(scratch, pc, nc * 8, 32);
            }
            r9[variant] = to_cycles(cyc() - t0) / nch;
            sme_stop();
        }
        printf("9. 预打包路径复现（每段周期，理想 %.0f）：无预取 %.0f (%.0f%%) | A 走线器 %.0f (%.0f%%) | A+X 走线器 %.0f (%.0f%%) | X 走线器 %.0f (%.0f%%) | A 驻留+X 流 %.0f (%.0f%%) | 全驻留 %.0f (%.0f%%) | A走线器+X驻留 %.0f (%.0f%%) | A走线器提前1条 %.0f (%.0f%%) | A+X走线器+预热 %.0f (%.0f%%) || C固定: 无预取 %.0f (%.0f%%) | A走线器 %.0f (%.0f%%) | A+X走线器 %.0f (%.0f%%)\n",
               ideal, r9[0], ideal / r9[0] * 100, r9[1], ideal / r9[1] * 100, r9[2], ideal / r9[2] * 100, r9[3], ideal / r9[3] * 100, r9[4], ideal / r9[4] * 100, r9[5], ideal / r9[5] * 100, r9[6], ideal / r9[6] * 100, r9[7], ideal / r9[7] * 100, r9[8], ideal / r9[8] * 100, r9[9], ideal / r9[9] * 100, r9[10], ideal / r9[10] * 100, r9[11], ideal / r9[11] * 100);
    }
    // 10. K 循环作为消费者：A 条（连续 16 KB）冷 / 走线器预取后 / 热 → 每步周期。B = Xp 热。
    {
        size_t stripd = (size_t)16 * kc;
        long nstr = 4096;                                              // 64 MB 的条池，保证冷
        double* PP2 = alloc_on(nstr * stripd * 8, -1);
        memset(PP2, 0, nstr * stripd * 8);
        double r10[6];
        const int nrep = 60;
        for (int v = 0; v < 6; v++) {
            double acc = 0;
            for (int rep = 0; rep < nrep; rep++) {
                const double* As = PP2 + (size_t)((rep * 7 + v * 500) % nstr) * stripd;
                sme_start();
                if (v == 1) {   // 走线器预取（4 个 K 循环，A/B 热）
                    for (int t = 0; t < 4; t++) { pf_t pf = { (const char*)As + (size_t)t * (kc / 2) * 64, 1L << 40, 0, (const char*)dummy, (const char*)dummy, 0 }; sme_zero_za(); kloop_plain(Ad, 128, 128, Xp, nc * 8, kc, 32, &pf); }
                } else if (v == 2) {   // 热：先跑一遍
                    sme_zero_za(); kloop_plain(As, 128, 128, Xp, nc * 8, kc, 32, &pfd);
                } else if (v == 3) {   // 突发 prfm
                    for (long b = 0; b < (long)stripd * 8; b += 64) asm volatile("prfm pldl1keep, [%0]" ::"r"((const char*)As + b));
                } else if (v == 4) {   // 突发真实装载
                    for (long b = 0; b < (long)stripd * 8; b += 64) asm volatile("ld1d z11.d, p0/z, [%0]" ::"r"((const char*)As + b) : "v11");
                } else if (v == 5) {   // 走线器预取，宿主 K 循环带冷 B（模拟真实：B 面板首条冷）
                    long chb = 300 + rep * 4;
                    for (int t = 0; t < 4; t++) { pf_t pf = { (const char*)As + (size_t)t * (kc / 2) * 64, 1L << 40, 0, (const char*)dummy, (const char*)dummy, 0 }; sme_zero_za(); kloop_plain(Ad, 128, 128, Xbig + (size_t)chb * kc * ldb + t * 32, ldb * 8, kc, 32, &pf); }
                }
                sme_zero_za();
                uint64_t tp = cyc();
                kloop_plain(As, 128, 128, Xp, nc * 8, kc, 32, &pfd);
                za_dump(scratch);
                acc += to_cycles(cyc() - tp);
                sme_stop();
            }
            r10[v] = acc / nrep;
        }
        printf("10. K 循环消费冷 A 条（%ld KB）：冷 %.0f | 走线器后 %.0f | 热 %.0f | 突发 prfm 后 %.0f | 突发装载后 %.0f | 走线器(宿主冷 B)后 %.0f  周期（理想 %ld）\n",
               stripd * 8 / 1024, r10[0], r10[1], r10[2], r10[3], r10[4], r10[5], kc * 8);
        // 10c. 连续消费 64 条 A（每条 4 个 K 循环，如真实结构）：顺序/随机 × 无预取/走线器(2 条前)/走线器(1 条前)
        {
            double r10c[6];
            for (int v = 0; v < 6; v++) {
                int seq = v < 3, mode = v % 3;
                double acc = 0;
                for (int rep = 0; rep < 8; rep++) {
                    long base = (rep * 811 + v * 3001) % (nstr - 200);
                    long idx[70];
                    for (int i = 0; i < 70; i++) idx[i] = seq ? base + i : (base + i * 37) % (nstr - 1);
                    sme_start();
                    uint64_t tp = cyc();
                    for (int i = 0; i < 64; i++) {
                        const double* As = PP2 + (size_t)idx[i] * stripd;
                        const double* An = PP2 + (size_t)idx[i + (mode == 2 ? 1 : 2)] * stripd;
                        for (int t = 0; t < 4; t++) {
                            pf_t pf = pfd;
                            if (mode) { pf.p = (const char*)An + (size_t)t * (kc / 2) * 64; pf.pfn = 1L << 40; pf.pfj = 0; }
                            sme_zero_za(); kloop_plain(As, 128, 128, Xp + 32 * t, nc * 8, kc, 32, &pf); za_dump(scratch);
                        }
                    }
                    acc += to_cycles(cyc() - tp) / 256;
                    sme_stop();
                }
                r10c[v] = acc / 8;
            }
            printf("10c. 连续 64 条×4 微块，每 K 循环周期（理想 %ld，热≈%.0f）：顺序: 无预取 %.0f | 走线器2前 %.0f | 走线器1前 %.0f || 随机: 无预取 %.0f | 走线器2前 %.0f | 走线器1前 %.0f\n",
                   kc * 8, r10[2], r10c[0], r10c[1], r10c[2], r10c[3], r10c[4], r10c[5]);
        }
        // 10b. 预取后隔 N 个"流式 K 循环"（各读一条冷 A + 回写 cbuf 微块）再消费：衰减曲线
        {
            double r10b[6]; int Ns[6] = {0, 2, 4, 8, 16, 32};
            for (int vi = 0; vi < 6; vi++) {
                double acc = 0;
                for (int rep = 0; rep < nrep; rep++) {
                    long base = (rep * 97 + vi * 7000) % (nstr - 64);
                    const double* As = PP2 + (size_t)base * stripd;
                    sme_start();
                    for (int t = 0; t < 4; t++) { pf_t pf = { (const char*)As + (size_t)t * (kc / 2) * 64, 1L << 40, 0, (const char*)dummy, (const char*)dummy, 0 }; sme_zero_za(); kloop_plain(Ad, 128, 128, Xp, nc * 8, kc, 32, &pf); }
                    for (int n = 0; n < Ns[vi]; n++) {   // 干扰：冷 A 条 + 宿主回写到 cbuf 的不同微块
                        const double* Ai = PP2 + (size_t)(base + 1 + n) * stripd;
                        sme_zero_za(); kloop_epi_sub_any(Ai, 128, 128, Xp, nc * 8, kc, 32, &pfd, scratch, C + (n % 64) * 32 * 16, nc * 8, 32); za_dump(scratch);
                    }
                    sme_zero_za();
                    uint64_t tp = cyc();
                    kloop_plain(As, 128, 128, Xp, nc * 8, kc, 32, &pfd);
                    za_dump(scratch);
                    acc += to_cycles(cyc() - tp);
                    sme_stop();
                }
                r10b[vi] = acc / nrep;
            }
            printf("10b. 预取后隔 N 个流式 K 循环再消费：N=0 %.0f | 2 %.0f | 4 %.0f | 8 %.0f | 16 %.0f | 32 %.0f 周期\n", r10b[0], r10b[1], r10b[2], r10b[3], r10b[4], r10b[5]);
        }
    }
    // 11. 对角阶段复现（全驻留）：D 条 + cbuf 都在 L2，逐条量 K 循环/转存/落地，与 run_task 阶段 3 相同的结构
    {
        long Sd = mb / 16;
        size_t dsz = diag_strip_off((int)Sd) * 8;
        double* D = alloc_on(dsz, -1);
        for (size_t q = 0; q < dsz / 8; q++) D[q] = 1e-3 * (q % 11);
        double* CB = alloc_on((size_t)mb * nc * 8, -1);
        for (long q = 0; q < mb * nc; q++) CB[q] = 1.0;
        long cs = nc * 8, ntile = nc / 32;
        long steps = 0; for (int sidx = 0; sidx < Sd; sidx++) steps += ntile * 16L * (sidx + 1);
        double best = 1e30, tk_sum = 0, te_sum = 0;
        for (int rep = 0; rep < 20; rep++) {
            sme_start();
            uint64_t t0 = cyc(); double tk = 0, te = 0;
            int pend = 0; double* pend_c = NULL; long pend_cols = 0;
            for (int sidx = 0; sidx < Sd; sidx++) {
                const double* A = D + diag_strip_off(sidx);
                long K = 16L * (sidx + 1);
                for (long t = 0; t < ntile; t++) {
                    pf_t pf = pfd;
                    if (sidx + 1 < Sd) pf.p2 = (const char*)(D + diag_strip_off(sidx + 1)) + t * K * 32;
                    pf.pc = (const char*)(CB + 16L * sidx * nc + 32 * t); pf.pcs = cs;
                    sme_zero_za();
                    uint64_t a0 = cyc();
                    if (pend && K >= 65) kloop_epi_set(A, 128, 128, CB + 32 * t, cs, K, 32, &pf, scratch, pend_c, cs, pend_cols);
                    else if (pend && K == 64) kloop_epi64_set(A, 128, 128, CB + 32 * t, cs, 32, &pf, scratch, pend_c, cs, pend_cols);
                    else { if (pend) epi_set_std(scratch, pend_c, cs, pend_cols); kloop_plain(A, 128, 128, CB + 32 * t, cs, K, 32, &pf); }
                    uint64_t a1 = cyc();
                    za_dump(scratch);
                    uint64_t a2 = cyc();
                    tk += to_cycles(a1 - a0); te += to_cycles(a2 - a1);
                    pend = 1; pend_c = CB + 16L * sidx * nc + 32 * t; pend_cols = 32;
                }
                if (pend) { uint64_t a0 = cyc(); epi_set_std(scratch, pend_c, cs, pend_cols); te += to_cycles(cyc() - a0); pend = 0; }
            }
            double tot = to_cycles(cyc() - t0);
            sme_stop();
            if (tot < best) { best = tot; tk_sum = tk; te_sum = te; }
        }
        printf("11. 对角阶段复现（S=%ld, ntile=%ld，全驻留）: 总 %.0f 周期 = %.2f 周期/步（理想 8.07，%ld 步，效率 %.1f%%）| K 循环 %.0f（%.2f/步）| 转存+落地 %.0f\n",
               Sd, ntile, best, best / steps, steps, 8.07 * steps / best * 100, tk_sum, tk_sum / steps, te_sum);
        // 11b. 同样的步数但只跑最长 K 循环（无转存），看短 K 循环的固定开销
        {
            sme_start();
            long K = 16L * Sd; long reps = steps / K;
            sme_zero_za();
            uint64_t t0 = cyc();
            for (long q = 0; q < reps; q++) kloop_plain(D, 128, 128, CB, cs, K, 32, &pfd);
            double tot = to_cycles(cyc() - t0);
            sme_stop();
            printf("11b. 等步数、只跑 K=%ld 的 kloop_plain（无转存）: %.2f 周期/步\n", K, tot / (reps * K));
            sme_start();
            K = 16; reps = steps / K;
            sme_zero_za();
            t0 = cyc();
            for (long q = 0; q < reps; q++) kloop_plain(D, 128, 128, CB, cs, K, 32, &pfd);
            tot = to_cycles(cyc() - t0);
            sme_stop();
            printf("11c. 等步数、只跑 K=16 的 kloop_plain（无转存）: %.2f 周期/步（每次调用固定开销 ≈ %.0f 周期）\n", tot / (reps * K), (tot / reps) - 16 * 8.07);
            sme_start();
            K = 64; reps = steps / K;
            t0 = cyc();
            for (long q = 0; q < reps; q++) { sme_zero_za(); kloop_plain(D, 128, 128, CB, cs, K, 32, &pfd); za_dump(scratch); }
            tot = to_cycles(cyc() - t0);
            sme_stop();
            printf("11d. K=64 kloop_plain + za_dump：%.0f 周期/次（K 循环 516 + 转存 98 = 614 为理想）\n", tot / reps);
            // 11e. 只 K 循环（不 zero 不转存，ZA 一直累加）
            sme_start(); t0 = cyc();
            for (long q = 0; q < reps; q++) kloop_plain(D, 128, 128, CB, cs, K, 32, &pfd);
            double te = to_cycles(cyc() - t0) / reps; sme_stop();
            // 11f. zero + K 循环
            sme_start(); t0 = cyc();
            for (long q = 0; q < reps; q++) { sme_zero_za(); kloop_plain(D, 128, 128, CB, cs, K, 32, &pfd); }
            double tf = to_cycles(cyc() - t0) / reps; sme_stop();
            // 11g. K 循环 + 转存（不 zero）
            sme_start(); t0 = cyc();
            for (long q = 0; q < reps; q++) { kloop_plain(D, 128, 128, CB, cs, K, 32, &pfd); za_dump(scratch); }
            double tg = to_cycles(cyc() - t0) / reps; sme_stop();
            // 11h. 转存后紧跟 K 循环（zero 放在转存前，看 zero 是否与 str 串行）
            sme_start(); t0 = cyc();
            for (long q = 0; q < reps; q++) { za_dump(scratch); sme_zero_za(); kloop_plain(D, 128, 128, CB, cs, K, 32, &pfd); }
            double th = to_cycles(cyc() - t0) / reps; sme_stop();
            // 11i. 只 za_dump（无计算）
            sme_start(); t0 = cyc();
            for (long q = 0; q < reps; q++) za_dump(scratch);
            double ti = to_cycles(cyc() - t0) / reps; sme_stop();
            printf("11e. K=64 只 kloop_plain 连续调用：%.0f 周期/次（理想 516，每次调用爬坡 ≈ %.0f）\n", te, te - 516);
            printf("11f. zero + kloop：%.0f | 11g. kloop + dump（不 zero）：%.0f | 11h. dump, zero, kloop：%.0f | 11i. 只 dump：%.0f\n", tf, tg, th, ti);
        }
    }

sec12:
    // 12. 半 ZA 交替原型（HZ）：16×16 微块，变体 A 算 za0-3、托管转存 za4-7→sB 与 sA→cbuf 的回写；变体 B 反之
    {
        long Kh = 64, ntile16 = nc / 16, cs = nc * 8;
        double* sA = alloc_on(4096, -1); double* sB = alloc_on(4096, -1);
        memset(sA, 0, 4096); memset(sB, 0, 4096);
        double* CB = alloc_on((size_t)16 * nc * 8, -1);
        double* CR = alloc_on((size_t)16 * nc * 8, -1);
        for (long q = 0; q < 16 * nc; q++) CB[q] = CR[q] = 1.0 + 1e-3 * (q % 7);
        fprintf(stderr, "[kb] sec12 start Kh=%ld ntile16=%ld\n", Kh, ntile16);
        // 参考：CR -= Ad(16×Kh, 128B/k) · Xp(Kh×nc)
        for (long k = 0; k < Kh; k++)
            for (int rr = 0; rr < 16; rr++)
                for (long j = 0; j < nc; j++) CR[rr * nc + j] -= Ad[k * 16 + rr] * Xp[k * nc + j];
        // 流水：tile t 算半 H(t&1)，转存 H(1-(t&1))（tile t-1）→ s[1-(t&1)]，回写 s[t&1]（tile t-2）→ CB(t-2)
        sme_start();
        sme_zero_za();
        uint64_t t0 = cyc();
        for (long t = 0; t < ntile16; t++) {
            double* xc = t >= 2 ? CB + 16 * (t - 2) : CB;            // t<2 时回写目标无所谓：ecols=0 → 谓词全假
            long ecols = t >= 2 ? 16 : 0;
            if ((t & 1) == 0) hz_kloop_A(Ad, Xp + 16 * t, cs, Kh, sB, sA, xc, cs, ecols, t >= 1);
            else              hz_kloop_B(Ad, Xp + 16 * t, cs, Kh, sA, sB, xc, cs, ecols, 1);
        }
        double tot = to_cycles(cyc() - t0);
        fprintf(stderr, "[kb] sec12 first pass done\n");
        // 收尾：转存最后一个 tile 的半，回写最后两个
        long tl = ntile16 - 1;
        if ((tl & 1) == 0) hz_dump_half0(sA); else hz_dump_half1(sB);
        sme_stop();
        hz_ev_std((tl & 1) ? sA : sB, CB + 16 * (tl - 1), cs, 16);   // tile tl-1 的半
        hz_ev_std((tl & 1) ? sB : sA, CB + 16 * tl, cs, 16);         // tile tl 的半
        double maxe = 0; for (long q = 0; q < 16 * nc; q++) { double e = fabs(CB[q] - CR[q]); if (e > maxe) maxe = e; }
        // 计时：多次重复整条（8 tile），比较当前 16×32 设计的 4×753
        double best = 1e30;
        for (int rep = 0; rep < 200; rep++) {
            sme_start(); sme_zero_za();
            uint64_t a0 = cyc();
            for (long t = 0; t < ntile16; t++) {
                if ((t & 1) == 0) hz_kloop_A(Ad, Xp + 16 * t, cs, Kh, sB, sA, CB + 16 * (t & 7), cs, 16, 1);
                else              hz_kloop_B(Ad, Xp + 16 * t, cs, Kh, sA, sB, CB + 16 * (t & 7), cs, 16, 1);
            }
            double c1 = to_cycles(cyc() - a0);
            sme_stop();
            if (c1 < best) best = c1;
        }
        printf("12. HZ 原型 16×16 半 ZA 交替，%ld tile × K=%ld：首轮 %.0f 周期，最好 %.0f 周期 = %.2f 周期/步（理想 4.0；等效 16×32 设计 4×753=3012）| 正确性 maxerr %.2e %s\n",
               ntile16, Kh, tot, best, best / (ntile16 * Kh), maxe, maxe < 1e-12 ? "ok" : "WRONG");
    }
    printf("5a. 流式 X 拷贝 + L JIT，无预取        : %8.0f 周期/段，效率 %.1f%%\n", r[9], ideal / r[9] * 100);
    printf("5b. 同上 + K 循环内预取 L 与 X         : %8.0f 周期/段，效率 %.1f%%\n", r[10], ideal / r[10] * 100);
    printf("5c. 同上 仅预取 X                      : %8.0f 周期/段，效率 %.1f%%\n", r[11], ideal / r[11] * 100);
    printf("5d. 仅 ld4 打包 %ld 条（源冷）           : %8.0f 周期/段 = %.0f 周期/条 = %.2f 周期/元素\n", S, r[12], r[12] / S, r[12] / S / (16 * kc));
    printf("5e. 仅 X 面板拷贝（冷）%ld KB           : %8.0f 周期/段 = %.1f GB/s\n", kc * nc * 8 / 1024, r[13], (double)kc * nc * 8 / (r[13] / 1.55e9) * 1e-9);
    return 0;
}
