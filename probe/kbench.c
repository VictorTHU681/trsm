// 单线程内核微基准 v2：include trsm.c 取用内部函数。流式模式只在测量段内进入。
// 编译：gcc -O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp kbench.c -o kbench -lm
// 运行：taskset -c 40 ./kbench [kc] [nc] [mb]
#include "../trsm.c"

static double to_cycles(uint64_t dt) { return (double)dt / (double)cyc_hz() * 1.55e9; }

int main(int argc, char** argv)
{
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
    pf_t pfd = { (const char*)dummy, 2, -128 };
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
    for (long k = 0; k < kc; k++) for (int rr = 0; rr < 16; rr++) if (A[k * 16 + rr] != Lsrc[rr * lda + k]) bad4++;
    t0 = cyc();
    for (int w = 0; w < reps; w++) pack_strip_ld2(Lsrc, lda, 0, 0, kc, A);
    r5v[1] = to_cycles(cyc() - t0) / reps;
    for (long k = 0; k < kc; k++) for (int rr = 0; rr < 16; rr++) { long off = (k & 1) ? 32 * k - 16 : 32 * k; if (A[off + rr] != Lsrc[rr * lda + k]) bad2++; }
    t0 = cyc();
    for (int w = 0; w < reps; w++) pack_strip_za(Lsrc, lda, 0, 0, kc, A);
    r5v[2] = to_cycles(cyc() - t0) / reps;
    for (long k = 0; k < kc; k++) for (int rr = 0; rr < 16; rr++) if (A[k * 64 + rr] != Lsrc[rr * lda + k]) badz++;
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
    pack_strip_ld4(Lsrc, lda, 0, 0, kc, A);   // 后续段用稠密布局
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
    printf("5a. 流式 X 拷贝 + L JIT，无预取        : %8.0f 周期/段，效率 %.1f%%\n", r[9], ideal / r[9] * 100);
    printf("5b. 同上 + K 循环内预取 L 与 X         : %8.0f 周期/段，效率 %.1f%%\n", r[10], ideal / r[10] * 100);
    printf("5c. 同上 仅预取 X                      : %8.0f 周期/段，效率 %.1f%%\n", r[11], ideal / r[11] * 100);
    return 0;
}
