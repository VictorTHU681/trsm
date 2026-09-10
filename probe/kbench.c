// 单线程内核微基准 v2：include trsm.c 取用内部函数。流式模式只在测量段内进入。
// 编译：gcc -O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp kbench.c -o kbench -lm
// 运行：taskset -c 40 ./kbench [kc] [nc] [mb]
#include "../trsm.c"
#include <signal.h>
#include <ucontext.h>

static double to_cycles(uint64_t dt) { return (double)dt / (double)cyc_hz() * 1.55e9; }

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
    sigaction(SIGILL, &sa, NULL); sigaction(SIGSEGV, &sa, NULL);
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
    pf_t pfd = { (const char*)dummy, 1, -64, (const char*)dummy };
    uint64_t t0;
    const int reps = 2000;
    double r[16];

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
    t0 = cyc();
    for (int w = 0; w < reps; w++) { sme_zero_za(); kloop_epi_sub(Ad, 128, 128, Xp, nc * 8, kc, 32, &pfd, scratch, C, nc * 8, 32); za_dump(scratch); }
    r[3] = to_cycles(cyc() - t0) / reps;
    t0 = cyc();
    for (int w = 0; w < reps; w++) { sme_zero_za(); kloop_epi_set(Ad, 128, 128, Xp, nc * 8, kc, 32, &pfd, scratch, C, nc * 8, 32); za_dump(scratch); }
    r[4] = to_cycles(cyc() - t0) / reps;
    sme_stop();
    double* tb2 = alloc_on(8192, -1); memset(tb2, 0, 8192);   // 注意：系统调用会退出流式模式，必须在 smstart 之前分配
    double* Anext = alloc_on(16 * kc * 8, -1); memset(Anext, 0, 16 * kc * 8);
    sme_start();
    t0 = cyc();
    for (int w = 0; w < reps; w++) { sme_zero_za(); kloop_tr_test(Ad, Xp, nc * 8, kc, tb2); }
    double rtr = to_cycles(cyc() - t0) / reps;
    // 真正的交错转置 K 循环（源 Ad 当作 16 行 L，行距 kc）：无回写 / 带回写
    t0 = cyc();
    for (int w = 0; w < reps; w++) { sme_zero_za(); kloop_tr_noepi(Ad, 128, 128, Xp, nc * 8, kc, &pfd, Ad, kc * 8, Anext); }
    double rtr2 = to_cycles(cyc() - t0) / reps;
    t0 = cyc();
    for (int w = 0; w < reps; w++) { sme_zero_za(); kloop_tr_epi(Ad, 128, 128, Xp, nc * 8, kc, &pfd, scratch, C, nc * 8, 32, Ad, kc * 8, Anext); za_dump(scratch); }
    double rtr3 = to_cycles(cyc() - t0) / reps;
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
                if (pend) kloop_epi_sub(A, 128, 128, Xp + 32 * t, nc * 8, kc, 32, &pfd, scratch, pc, nc * 8, 32);
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
                if (pend) kloop_epi_sub(A, 128, 128, Xp + 32 * t, nc * 8, kc, 32, &pfd, scratch, pc, nc * 8, 32);
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
                    if (pend) kloop_epi_sub(A, 128, 128, Xpan + 32 * t, nc * 8, kc, 32, &pf, scratch, pc, nc * 8, 32);
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
                        pf_t pf = { (const char*)(Xk + q * 8 * ldb), xlines, ldb * 8 - xlines * 64, (const char*)dummy };
                        sme_zero_za(); kloop_plain(A, 128, 128, Xp, nc * 8, kc, 32, &pf);
                    }
                } else if (v == 2) {   // 突发 prfm pldl1keep
                    for (long k = 0; k < kc; k++) for (long j = 0; j < nc; j += 8) asm volatile("prfm pldl1keep, [%0]" ::"r"(Xk + k * ldb + j));
                } else if (v == 3) {   // 突发 prfm pldl2keep
                    for (long k = 0; k < kc; k++) for (long j = 0; j < nc; j += 8) asm volatile("prfm pldl2keep, [%0]" ::"r"(Xk + k * ldb + j));
                } else if (v == 4) {   // 真实装载触碰（每行 16 条 ld1d 到哑寄存器）
                    for (long k = 0; k < kc; k++) for (long j = 0; j < nc; j += 8) asm volatile("ld1d z31.d, p0/z, [%0]" ::"r"(Xk + k * ldb + j) : "v31");
                } else if (v == 5) {   // K 循环走线器，但每步 2 行连续（把面板当成连续 128 KB：pfn 极大，pfj 0）—— 检验连续目标
                    pf_t pf = { (const char*)Xpan, 1L << 40, 0, (const char*)dummy };
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
    printf("5a. 流式 X 拷贝 + L JIT，无预取        : %8.0f 周期/段，效率 %.1f%%\n", r[9], ideal / r[9] * 100);
    printf("5b. 同上 + K 循环内预取 L 与 X         : %8.0f 周期/段，效率 %.1f%%\n", r[10], ideal / r[10] * 100);
    printf("5c. 同上 仅预取 X                      : %8.0f 周期/段，效率 %.1f%%\n", r[11], ideal / r[11] * 100);
    printf("5d. 仅 ld4 打包 %ld 条（源冷）           : %8.0f 周期/段 = %.0f 周期/条 = %.2f 周期/元素\n", S, r[12], r[12] / S, r[12] / S / (16 * kc));
    printf("5e. 仅 X 面板拷贝（冷）%ld KB           : %8.0f 周期/段 = %.1f GB/s\n", kc * nc * 8 / 1024, r[13], (double)kc * nc * 8 / (r[13] / 1.55e9) * 1e-9);
    return 0;
}
