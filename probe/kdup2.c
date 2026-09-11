// 微基准：双 bank 内核在“真实访问模式”下的每 k 步周期数——比较不同 (S, NT, C 行距) 阻塞下的单核 K 循环效率。
//   每“段”S 条 A（各 8 KB，连续）× NT 个 B 微面板（各 16 KB）× C 块 S 条 × NT 微块（行距 cs 字节，常驻）。
//   A 段、B 段流式换（大缓冲）；A 由 ap 游标预取下一条（每条恰好 8 KB：t=0 用全预取内核，t≥1 用 _ah 内核），
//   B 由 bp 游标在段的最后 8 条里预取下一段（每微块调用 2 KB）。
// 编译：gcc -O2 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp probe/kdup2.c -o probe/kdup2 -lm
// 运行：numactl -N 1 -m 17 taskset -c 40 ./probe/kdup2
#include "../trsm.c"

static double* al(size_t bytes) { void* p = aligned_alloc(4096, bytes); memset(p, 0, bytes); return p; }
static const long K = 64;

typedef struct { double *A, *B, *C; size_t abytes, bbytes; long S, NT; long cs; int ah; int fixed; } cfg_t;

static void kcall(const cfg_t* g, const double* A, const double* B, double* Cprev, double* C, const char* ap, const char* bp, long apoff4)
{
    const char* pc = (const char*)C;
    if (g->cs == 1792) {
        if (g->ah) {
            kloop_duplex_bank0_c224_ah(A, B, (Cprev ? Cprev : C) + 16, ap, bp, pc);
            kloop_duplex_bank4_c224_ah(A, B + 1024, C, ap + apoff4, bp + 1024, pc + 128);
        } else {
            kloop_duplex_bank0_c224(A, B, (Cprev ? Cprev : C) + 16, ap, bp, pc);
            kloop_duplex_bank4_c224(A, B + 1024, C, ap + apoff4, bp + 1024, pc + 128);
        }
    } else {
        if (g->ah) {
            kloop_duplex_bank0_ah(A, B, (Cprev ? Cprev : C) + 16, ap, bp, pc);
            kloop_duplex_bank4_ah(A, B + 1024, C, ap + apoff4, bp + 1024, pc + 128);
        } else {
            kloop_duplex_bank0(A, B, (Cprev ? Cprev : C) + 16, ap, bp, pc);
            kloop_duplex_bank4(A, B + 1024, C, ap + apoff4, bp + 1024, pc + 128);
        }
    }
}

static double run(const cfg_t* g, long nchunk, char* dummy)
{
    const long S = g->S, NT = g->NT, ncp = g->cs / 8;
    const size_t achunk = S * 1024 * 8, bchunk = NT * 2048 * 8;
    const long na = g->fixed ? 1 : g->abytes / achunk, nb = g->fixed ? 1 : g->bbytes / bchunk;
    const long calls = S * NT;
    const long bwin = 8 * NT;                 // B 预取窗口：最后 8 条的调用数，每调用 2 KB
    uint64_t t0 = cyc();
    double* pend_c = NULL;
    for (long k = 0; k < nchunk; k++) {
        const double* Ach = g->A + (k % na) * (achunk / 8);
        const double* Anext = g->A + ((k + 1) % na) * (achunk / 8);
        const double* Bch = g->B + (k % nb) * (bchunk / 8);
        const double* Bnext = g->B + ((k + 1) % nb) * (bchunk / 8);
        long call = 0;
        for (long s = 0; s < S; s++) {
            const double* A = Ach + s * 1024;
            const double* An = (s + 1 < S) ? Ach + (s + 1) * 1024 : Anext;
            for (long t = 0; t < NT; t++, call++) {
                const double* B = Bch + t * 2048;
                double* C = g->C + 16 * s * ncp + 32 * t;
                const char* ap; long ap4; cfg_t gg = *g;
                if (g->fixed) { ap = dummy; ap4 = 1024; gg.ah = 0; }
                else if (NT == 4 || !g->ah) { ap = (const char*)An + t * (8192 / NT); ap4 = 1024; gg.ah = 0; }
                else if (t == 0) { ap = (const char*)An; ap4 = 1024; gg.ah = 0; }
                else { ap = (const char*)An + 2048 + (t - 1) * 1024; ap4 = 512; gg.ah = 1; }
                const char* bp = dummy;
                long bc = call - (calls - bwin);
                if (!g->fixed && bc >= 0) bp = (const char*)Bnext + (size_t)bc * 2048;
                kcall(&gg, A, B, pend_c, C, ap, bp, ap4);
                pend_c = C;
            }
        }
    }
    uint64_t t1 = cyc();
    return (double)(t1 - t0) / (nchunk * calls) / K * (1.55e9 / (double)cyc_hz());
}

static const char* names[32]; static double vals[32]; static int nres = 0;
static void show(const char* name, double r) { names[nres] = name; vals[nres++] = r; }   // 流式模式下不能 printf

int main(void)
{
    char* dummy = (char*)al(1 << 20);
    const size_t big = 256UL << 20;
    double* Abig = al(big); double* Bbig = al(big); double* Cbig = al(8UL << 20);
    for (size_t i = 0; i < big / 8; i += 7) Abig[i] = 1.0 / (i + 1), Bbig[i] = 1.0 / (i + 3);
    sme_start();
    const long n = 300;
    cfg_t v;
    cfg_t cur = { Abig, Bbig, Cbig, big, big, 16, 4, 1024, 0, 0 };
    v = cur; v.fixed = 1; run(&v, 4, dummy); show("S16 NT4 cs1024 全 L2 驻留", run(&v, n, dummy));
    v = cur; show("S16 NT4 cs1024（当前 case2）", run(&v, n, dummy));
    v = cur; v.S = 12; show("S12 NT4 cs1024（当前 case3 形状）", run(&v, n, dummy));
    v = cur; v.S = 12; v.NT = 7; v.cs = 1792; v.ah = 1; v.fixed = 1; run(&v, 4, dummy); show("S12 NT7 cs1792 全 L2 驻留", run(&v, n, dummy));
    v = cur; v.S = 12; v.NT = 7; v.cs = 1792; v.ah = 1; show("S12 NT7 cs1792 ah（候选 mb192/nc224）", run(&v, n, dummy));
    v = cur; v.S = 12; v.NT = 7; v.cs = 1792; v.ah = 0; show("S12 NT7 cs1792 全预取（A 超预取）", run(&v, n, dummy));
    v = cur; v.S = 16; v.NT = 7; v.cs = 1792; v.ah = 1; show("S16 NT7 cs1792 ah（C 448 KB）", run(&v, n, dummy));
    v = cur; v.S = 8; v.NT = 7; v.cs = 1792; v.ah = 1; show("S8 NT7 cs1792 ah（C 224 KB）", run(&v, n, dummy));
    v = cur; v.S = 12; v.NT = 7; v.cs = 1792; v.ah = 1; show("S12 NT7 cs1792 ah（重复）", run(&v, n, dummy));
    v = cur; show("S16 NT4 cs1024（当前，重复）", run(&v, n, dummy));
    sme_stop();
    printf("每 16×32 等效 k 步周期数（理想 8）：\n");
    for (int i = 0; i < nres; i++) printf("%-44s: %6.2f  效率 %.1f%%\n", names[i], vals[i], 800 / vals[i]);
    return 0;
}
