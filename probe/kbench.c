// 单线程内核微基准：直接 include trsm.c 取用内部函数，逐项测 K 循环 / ZA 转存+回写 / 打包 / 整段 的周期数。
// 编译：gcc -O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp kbench.c -o kbench -lm
// 运行：taskset -c 40 ./kbench [kc] [nc] [mb]
#include "../trsm.c"

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// 以 cntvct 计时换算成核周期（1.55 GHz）
static double to_cycles(uint64_t dt) { return (double)dt / (double)cyc_hz() * 1.55e9; }

int main(int argc, char** argv)
{
    long kc = argc > 1 ? atol(argv[1]) : 128;
    long nc = argc > 2 ? atol(argv[2]) : 128;
    long mb = argc > 3 ? atol(argv[3]) : 256;
    long S = mb / 16, ntile = nc / 32;
    printf("kc=%ld nc=%ld mb=%ld  cpu=%d  cntfrq=%lu\n", kc, nc, mb, sched_getcpu(), (unsigned long)cyc_hz());

    double* A = alloc_on(16 * kc * 8, -1);          // 打包 A 条
    double* Xp = alloc_on(kc * nc * 8, -1);         // X 面板（kc × nc，行距 nc）
    double* C = alloc_on(mb * nc * 8, -1);          // C 块
    double* scratch = alloc_on(8192, -1);
    for (long i = 0; i < 16 * kc; i++) A[i] = 1e-3 * (i % 17);
    for (long i = 0; i < kc * nc; i++) Xp[i] = 1e-3 * (i % 13);
    for (long i = 0; i < mb * nc; i++) C[i] = 1.0;

    sme_start();
    // 1. 纯 K 循环（A、B 都在 L1/L2）
    {
        for (int w = 0; w < 10; w++) { sme_zero_za(); kloop_16x32(A, Xp, nc * 8, kc, 32, A, 0); }
        int reps = 2000;
        uint64_t t0 = cyc();
        for (int w = 0; w < reps; w++) { sme_zero_za(); kloop_16x32(A, Xp, nc * 8, kc, 32, A, 0); }
        double cyc_per = to_cycles(cyc() - t0) / reps;
        printf("1. kloop K=%ld（驻留）        : %8.0f 周期 = %.2f 周期/步（理想 8）\n", kc, cyc_per, cyc_per / kc);
    }
    // 2. K 循环 + ZA 转存 + epi_sub
    {
        int reps = 2000;
        uint64_t t0 = cyc();
        for (int w = 0; w < reps; w++) { sme_zero_za(); kloop_16x32(A, Xp, nc * 8, kc, 32, A, 0); za_dump(scratch); epi_sub(scratch, C, nc * 8, 32); }
        double cyc_per = to_cycles(cyc() - t0) / reps;
        printf("2. kloop + dump + epi_sub        : %8.0f 周期 → 转存+回写 ≈ %.0f 周期/微块\n", cyc_per, cyc_per - kc * 8.0);
        t0 = cyc();
        for (int w = 0; w < reps; w++) { za_dump(scratch); }
        printf("   仅 za_dump                    : %8.0f 周期\n", to_cycles(cyc() - t0) / reps);
        t0 = cyc();
        for (int w = 0; w < reps; w++) { epi_sub(scratch, C, nc * 8, 32); }
        printf("   仅 epi_sub                    : %8.0f 周期\n", to_cycles(cyc() - t0) / reps);
        t0 = cyc();
        for (int w = 0; w < reps; w++) { sme_zero_za(); }
        printf("   仅 zero za                    : %8.0f 周期\n", to_cycles(cyc() - t0) / reps);
    }
    // 3. 打包：16 行 × kc（L2 驻留源）
    {
        long lda = 17024 + 64;
        double* Lsrc = alloc_on(16 * lda * 8 + kc * 8, -1);
        for (long i = 0; i < 16 * lda; i++) Lsrc[i] = 1e-3 * (i % 7);
        for (int w = 0; w < 10; w++) pack_strip(Lsrc, lda, 1 << 30, 0, 0, kc, A);
        int reps = 2000;
        uint64_t t0 = cyc();
        for (int w = 0; w < reps; w++) pack_strip(Lsrc, lda, 1 << 30, 0, 0, kc, A);
        double cyc_per = to_cycles(cyc() - t0) / reps;
        printf("3. pack_strip 16×%ld（源驻留）    : %8.0f 周期 = %.2f 周期/元素（K 循环占 %ld×%ld×8=%ld）\n", kc, cyc_per, cyc_per / (16 * kc), ntile, kc, ntile * kc * 8);
        // 标量路径对照（rows=15）
        t0 = cyc();
        for (int w = 0; w < reps; w++) pack_strip(Lsrc, lda, 15, 0, 0, kc, A);
        printf("   标量路径（15 行）             : %8.0f 周期\n", to_cycles(cyc() - t0) / reps);
    }
    // 4. 整段：S 条 × ntile 微块（X 面板驻留、A 条驻留）
    {
        int reps = 20;
        uint64_t t0 = cyc();
        for (int w = 0; w < reps; w++)
            for (long r = 0; r < S; r++)
                for (long t = 0; t < ntile; t++) {
                    sme_zero_za();
                    kloop_16x32(A, Xp + 32 * t, nc * 8, kc, 32, A, 0);
                    za_dump(scratch);
                    epi_sub(scratch, C + 16 * r * nc + 32 * t, nc * 8, 32);
                }
        double cyc_per = to_cycles(cyc() - t0) / reps;
        double ideal = (double)S * ntile * kc * 8;
        printf("4. 整段 %ld 条×%ld 块（全驻留）      : %8.0f 周期，理想 %.0f，效率 %.1f%%\n", S, ntile, cyc_per, ideal, ideal / cyc_per * 100);
    }
    // 5. 整段 + 流式：X 面板从大数组（HBM/DDR）取、L 行 JIT 打包（源为大数组），C 块驻留
    {
        long m = 17024, ldb = 512 + 64, lda = m + 64;
        double* Xbig = alloc_on((size_t)m * ldb * 8, -1);
        double* Lbig = alloc_on((size_t)m * lda * 8, -1);
        memset(Xbig, 0, (size_t)m * ldb * 8);
        memset(Lbig, 0, (size_t)m * lda * 8);
        int nchunk = (int)(m / kc) - 2;
        // 5a：只流 X（A 条驻留）
        uint64_t t0 = cyc();
        for (int ch = 0; ch < nchunk; ch++) {
            const double* Xk = Xbig + (size_t)ch * kc * ldb;
            for (long r = 0; r < S; r++)
                for (long t = 0; t < ntile; t++) {
                    sme_zero_za();
                    kloop_16x32(A, Xk + 32 * t, ldb * 8, kc, 32, A, 0);
                    za_dump(scratch);
                    epi_sub(scratch, C + 16 * r * nc + 32 * t, nc * 8, 32);
                }
        }
        double cyc_per = to_cycles(cyc() - t0) / nchunk;
        double ideal = (double)S * ntile * kc * 8;
        printf("5a. 每段 X 流式(无预取) A 驻留     : %8.0f 周期，效率 %.1f%%\n", cyc_per, ideal / cyc_per * 100);
        // 5b：X 流式 + 下一段 X 预取（分摊到每条）
        t0 = cyc();
        for (int ch = 0; ch < nchunk; ch++) {
            const double* Xk = Xbig + (size_t)ch * kc * ldb;
            long pfr = (kc + S - 1) / S;
            for (long r = 0; r < S; r++) {
                long pr0 = r * pfr, prn = pfr; if (pr0 + prn > kc) prn = kc - pr0;
                if (prn > 0) prefetch_rows(Xbig + (size_t)(ch + 1) * kc * ldb + pr0 * ldb, prn, ldb * 8, nc * 8);
                for (long t = 0; t < ntile; t++) {
                    sme_zero_za();
                    kloop_16x32(A, Xk + 32 * t, ldb * 8, kc, 32, A, 0);
                    za_dump(scratch);
                    epi_sub(scratch, C + 16 * r * nc + 32 * t, nc * 8, 32);
                }
            }
        }
        cyc_per = to_cycles(cyc() - t0) / nchunk;
        printf("5b. 每段 X 流式(+预取) A 驻留      : %8.0f 周期，效率 %.1f%%\n", cyc_per, ideal / cyc_per * 100);
        // 5c：X 流式+预取，L JIT 打包（源流式 + 下一条预取）
        t0 = cyc();
        for (int ch = 0; ch < nchunk; ch++) {
            const double* Xk = Xbig + (size_t)ch * kc * ldb;
            long k0 = (size_t)ch * kc;
            long pfr = (kc + S - 1) / S;
            for (long r = 0; r < S; r++) {
                pack_strip(Lbig, lda, m, 8192 + 16 * r, k0, kc, A);
                long nr = (r + 1 < S) ? 8192 + 16 * (r + 1) : 8192;
                long nk = (r + 1 < S) ? k0 : k0 + kc;
                prefetch_rows(Lbig + nr * lda + nk, 16, lda * 8, kc * 8);
                long pr0 = r * pfr, prn = pfr; if (pr0 + prn > kc) prn = kc - pr0;
                if (prn > 0) prefetch_rows(Xbig + (size_t)(ch + 1) * kc * ldb + pr0 * ldb, prn, ldb * 8, nc * 8);
                for (long t = 0; t < ntile; t++) {
                    sme_zero_za();
                    kloop_16x32(A, Xk + 32 * t, ldb * 8, kc, 32, A, 0);
                    za_dump(scratch);
                    epi_sub(scratch, C + 16 * r * nc + 32 * t, nc * 8, 32);
                }
            }
        }
        cyc_per = to_cycles(cyc() - t0) / nchunk;
        printf("5c. 每段 X 流式+预取, L JIT 打包   : %8.0f 周期，效率 %.1f%%\n", cyc_per, ideal / cyc_per * 100);
        // 5d：同 5c 但不预取
        t0 = cyc();
        for (int ch = 0; ch < nchunk; ch++) {
            const double* Xk = Xbig + (size_t)ch * kc * ldb;
            long k0 = (size_t)ch * kc;
            for (long r = 0; r < S; r++) {
                pack_strip(Lbig, lda, m, 8192 + 16 * r, k0, kc, A);
                for (long t = 0; t < ntile; t++) {
                    sme_zero_za();
                    kloop_16x32(A, Xk + 32 * t, ldb * 8, kc, 32, A, 0);
                    za_dump(scratch);
                    epi_sub(scratch, C + 16 * r * nc + 32 * t, nc * 8, 32);
                }
            }
        }
        cyc_per = to_cycles(cyc() - t0) / nchunk;
        printf("5d. 同 5c 无预取                   : %8.0f 周期，效率 %.1f%%\n", cyc_per, ideal / cyc_per * 100);
    }
    sme_stop();
    (void)now_s;
    return 0;
}
