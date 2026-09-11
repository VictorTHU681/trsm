// 微基准：双 bank 内核在“真实访问模式”下的每 k 步周期数——把 K 循环 69% 效率的损失定位到 L1 / L2 / 内存哪一层。
//   模式：每“段”16 条 A（各 8 KB，连续）× 4 个 B 微面板（各 16 KB）× C 块 16 条 × 4 微块（行距 1 KB，256 KB）。
//   v1 L1 驻留（同一 A/B/C 反复）；v2 全部在 L2（A 段 128 KB + B 段 64 KB + C 256 KB 固定）；
//   v3 A 段流式换（大缓冲，带 ap 预取到 L2）；v4 A、B 段都流式换（B 由 bp 预取）；v5 同 v4 但 C 块也换（每段一个 C 块）。
// 编译：gcc -O2 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp probe/kdup2.c -o probe/kdup2 -lm
// 运行：numactl -N 1 -m 17 taskset -c 40 ./probe/kdup2
#include "../trsm.c"

static double* al(size_t bytes) { void* p = aligned_alloc(4096, bytes); memset(p, 0, bytes); return p; }
static const long K = 64, S = 16, NT = 4;

// 跑 nchunk 个段；A 段地址由 achunk(k)、B 段由 bchunk(k)、C 块由 cblk(k) 给出
typedef struct { double *A, *B, *C; size_t abytes, bbytes, cbytes; int pfa, pfb; } cfg_t;
static double run(const cfg_t* g, long nchunk, char* dummy)
{
    const size_t achunk = S * 1024 * 8, bchunk = NT * 2048 * 8, cblk = 16L * S * 128 * 8;
    const long na = g->abytes / achunk, nb = g->bbytes / bchunk, ncb = g->cbytes / cblk;
    uint64_t t0 = cyc();
    double* pend_c = NULL;
    for (long k = 0; k < nchunk; k++) {
        const double* Ach = g->A + (k % na) * (achunk / 8);
        const double* Anext = g->A + ((k + 1) % na) * (achunk / 8);
        const double* Bch = g->B + (k % nb) * (bchunk / 8);
        const double* Bnext = g->B + ((k + 1) % nb) * (bchunk / 8);
        double* Cb = g->C + (k % ncb) * (cblk / 8);
        long call = 0;
        for (long s = 0; s < S; s++) {
            const double* A = Ach + s * 1024;
            const double* An = (s + 1 < S) ? Ach + (s + 1) * 1024 : Anext;
            for (long t = 0; t < NT; t++, call++) {
                const double* B = Bch + t * 2048;
                double* C = Cb + 16 * s * 128 + 32 * t;
                const char* ap = g->pfa ? (const char*)An + t * 2048 : dummy;
                const char* bp = (g->pfb && call < 32) ? (const char*)Bnext + call * 2048 : dummy;
                const char* pc = (const char*)C;
                kloop_duplex_bank0(A, B, (pend_c ? pend_c : C) + 16, ap, bp, pc);
                kloop_duplex_bank4(A, B + 1024, C, ap + 1024, bp + 1024, pc + 128);
                pend_c = C;
            }
        }
    }
    uint64_t t1 = cyc();
    return (double)(t1 - t0) / (nchunk * S * NT) / K * (1.55e9 / (double)cyc_hz());
}

int main(void)
{
    char* dummy = (char*)al(1 << 20);
    const size_t big = 256UL << 20;
    double* Abig = al(big); double* Bbig = al(big); double* Cbig = al(64UL << 20);
    for (size_t i = 0; i < big / 8; i += 7) Abig[i] = 1.0 / (i + 1), Bbig[i] = 1.0 / (i + 3);
    sme_start();
    cfg_t v1 = { Abig, Bbig, Cbig, 1024 * 8 * S, 2048 * 8 * NT, 16L * S * 128 * 8, 0, 0 };   // 单段，反复
    // v1 严格 L1：把段缩到一条 A、一块 B（同 kdup.c）
    double r1;
    {
        uint64_t t0 = cyc();
        for (long r = 0; r < 20000; r++) {
            kloop_duplex_bank0(Abig, Bbig, Cbig + 16, dummy, dummy, dummy);
            kloop_duplex_bank4(Abig, Bbig + 1024, Cbig, dummy + 1024, dummy + 1024, dummy + 128);
        }
        r1 = (double)(cyc() - t0) / 20000 / K * (1.55e9 / (double)cyc_hz());
    }
    cfg_t v2 = v1;                                                                       // 全 L2 驻留：A 128 KB + B 64 KB + C 256 KB
    cfg_t v3 = v1; v3.abytes = big; v3.pfa = 1;                                          // A 流式（预取）
    cfg_t v3n = v3; v3n.pfa = 0;                                                         // A 流式（不预取）
    cfg_t v4 = v3; v4.bbytes = big; v4.pfb = 1;                                          // A、B 流式
    cfg_t v5 = v4; v5.cbytes = 64UL << 20;                                               // C 块也换
    cfg_t v6 = v2; v6.cbytes = 64UL << 20;                                               // 只 C 块换（A/B 在 L2）
    long n = 400;
    run(&v2, 8, dummy);
    double r2 = run(&v2, n, dummy), r3 = run(&v3, n, dummy), r3n = run(&v3n, n, dummy), r4 = run(&v4, n, dummy), r5 = run(&v5, n, dummy), r6 = run(&v6, n, dummy);
    sme_stop();
    printf("每 16×32 等效 k 步周期数（理想 8）：\n");
    printf("v1 L1 驻留（同 A/B/C）          : %6.2f  效率 %.1f%%\n", r1, 800 / r1);
    printf("v2 L2 驻留（A128K+B64K+C256K）  : %6.2f  效率 %.1f%%\n", r2, 800 / r2);
    printf("v3 A 流式 + ap 预取             : %6.2f  效率 %.1f%%\n", r3, 800 / r3);
    printf("v3n A 流式 无预取               : %6.2f  效率 %.1f%%\n", r3n, 800 / r3n);
    printf("v4 A、B 流式（ap、bp 预取）      : %6.2f  效率 %.1f%%\n", r4, 800 / r4);
    printf("v5 A、B、C 全换                 : %6.2f  效率 %.1f%%\n", r5, 800 / r5);
    printf("v6 只 C 块换（A/B L2）           : %6.2f  效率 %.1f%%\n", r6, 800 / r6);
    return 0;
}
