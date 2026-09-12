// 微基准：双 bank 内核在“真实访问模式”下的每 k 步周期数——把 K 循环 69% 效率的损失定位到 L1 / L2 / 内存哪一层。
//   模式：每“段”S 条 A（各 8 KB，连续）× NT 个 B 微面板（各 16 KB）× C 块 S 条 × NT 微块（行距 1 KB）。
//   v1 L1 驻留；v2 全部在 L2；v3 A 段流式换（ap 预取到 L2）；v4 A、B 段都流式换（bp 预取）；再对 v4 做变体定位原因。
//   （内核提示变体 ldnt1d / 半预取 / pldl1keep / pldl2strm 在 fresh2 分支的版本里，结论见 NOTES §10b。）
// 编译：gcc -O2 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp probe/kdup2.c -o probe/kdup2 -lm
// 运行：numactl -N 1 -m 17 taskset -c 40 ./probe/kdup2
#include "../trsm.c"

static double* al(size_t bytes) { void* p = aligned_alloc(4096, bytes); memset(p, 0, bytes); return p; }
static const long K = 64;

typedef struct { double *A, *B, *C; size_t abytes, bbytes; long S, NT, cS; int pfa, pfb, bspread, adist, blate; } cfg_t;
static double run(const cfg_t* g, long nchunk, char* dummy)
{
    const long S = g->S, NT = g->NT;
    const size_t achunk = S * 1024 * 8, bchunk = NT * 2048 * 8;
    const long na = g->abytes / achunk, nb = g->bbytes / bchunk;
    const long calls = S * NT;
    const long bcalls = g->bspread ? calls : (bchunk / 2048);   // 预取下一 B 段所需的调用数（每调用 2 KB）
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
            const long ad = g->adist ? g->adist : 1;
            const double* An = (s + ad < S) ? Ach + (s + ad) * 1024 : Anext + (s + ad - S) * 1024;
            for (long t = 0; t < NT; t++, call++) {
                const double* B = Bch + t * 2048;
                double* C = g->C + 16 * (s % g->cS) * 128 + 32 * t;
                const char* ap = g->pfa ? (const char*)An + t * (8192 / NT) : dummy;
                const char* bp = dummy;
                long bc = g->blate ? call - (calls - bcalls) : call;
                if (g->pfb && bc >= 0 && bc < bcalls) {
                    size_t off = g->bspread ? (size_t)bc * (bchunk / calls) : (size_t)bc * 2048;
                    bp = (const char*)Bnext + off;
                }
                const char* pc = (const char*)C;
                kloop_duplex_bank0(A, B, (pend_c ? pend_c : C) + 16, ap, bp, pc);
                kloop_duplex_bank4(A, B + 1024, C, ap + 1024, bp + 1024, pc + 128);
                pend_c = C;
            }
        }
    }
    uint64_t t1 = cyc();
    return (double)(t1 - t0) / (nchunk * calls) / K * (1.55e9 / (double)cyc_hz());
}

static const char* names[32]; static double vals[32]; static int nres = 0;
static void show(const char* name, double r) { names[nres] = name; vals[nres++] = r; }   // 流式模式下不能 printf（syscall 清 SM → SIGILL）

int main(void)
{
    char* dummy = (char*)al(1 << 20);
    const size_t big = 256UL << 20;
    double* Abig = al(big); double* Bbig = al(big); double* Cbig = al(8UL << 20);
    for (size_t i = 0; i < big / 8; i += 7) Abig[i] = 1.0 / (i + 1), Bbig[i] = 1.0 / (i + 3);
    sme_start();
    double r1;
    {
        uint64_t t0 = cyc();
        for (long r = 0; r < 20000; r++) {
            kloop_duplex_bank0(Abig, Bbig, Cbig + 16, dummy, dummy, dummy);
            kloop_duplex_bank4(Abig, Bbig + 1024, Cbig, dummy + 1024, dummy + 1024, dummy + 128);
        }
        r1 = (double)(cyc() - t0) / 20000 / K * (1.55e9 / (double)cyc_hz());
    }
    show("v1 L1 驻留（同 A/B/C）", r1);
    cfg_t base = { Abig, Bbig, Cbig, 16 * 1024 * 8, 4 * 2048 * 8, 16, 4, 16, 0, 0, 0, 1, 0 };
    const long n = 400;
    cfg_t v; 
    v = base; run(&v, 8, dummy); show("v2 L2 驻留（A128K+B64K+C256K）", run(&v, n, dummy));
    v = base; v.abytes = big; v.pfa = 1; show("v3 A 流式 + ap 预取", run(&v, n, dummy));
    v = base; v.abytes = big; v.pfa = 1; v.bbytes = big; v.pfb = 1; show("v4 A、B 流式（ap、bp 预取，前 32 调用）", run(&v, n, dummy));
    v = base; v.abytes = big; v.pfa = 1; v.bbytes = big; v.pfb = 0; show("v4n A、B 流式，B 不预取", run(&v, n, dummy));
    v = base; v.abytes = big; v.pfa = 1; v.bbytes = big; v.pfb = 1; v.bspread = 1; show("v4s B 预取铺满 64 调用", run(&v, n, dummy));
    v = base; v.abytes = big; v.pfa = 1; v.bbytes = big; v.pfb = 1; v.cS = 4; show("v4c C 块只 64 KB（容量测试）", run(&v, n, dummy));
    v = base; v.abytes = big; v.pfa = 1; v.bbytes = big; v.pfb = 1; v.cS = 1; show("v4c1 C 块只 16 KB", run(&v, n, dummy));
    v = base; v.abytes = big; v.pfa = 1; v.bbytes = big; v.pfb = 1; v.S = 8; v.cS = 8; show("v4s8 S=8（A 段 64K，C 128K）", run(&v, n, dummy));
    v = base; v.abytes = big; v.pfa = 1; v.bbytes = big; v.pfb = 1; v.NT = 2; show("v4t2 NT=2（B 段 32K）", run(&v, n, dummy));
    v = base; v.abytes = big; v.pfa = 0; v.bbytes = big; v.pfb = 1; show("v4a A 不预取、B 预取", run(&v, n, dummy));
    v = base; v.bbytes = big; v.pfb = 1; show("v5 只 B 流式（A 在 L2）+ bp 预取", run(&v, n, dummy));
    v = base; v.bbytes = big; v.pfb = 0; show("v5n 只 B 流式，不预取", run(&v, n, dummy));
    v = base; v.abytes = big; v.pfa = 1; v.bbytes = big; v.pfb = 1; v.adist = 2; show("v4d2 A 预取提前 2 条", run(&v, n, dummy));
    v = base; v.abytes = big; v.pfa = 1; v.bbytes = big; v.pfb = 1; v.adist = 4; show("v4d4 A 预取提前 4 条", run(&v, n, dummy));
    v = base; v.abytes = big; v.pfa = 1; v.bbytes = big; v.pfb = 1; v.blate = 1; show("v4bl B 预取放在后 32 调用", run(&v, n, dummy));
    v = base; v.abytes = big; v.pfa = 1; v.bbytes = big; v.pfb = 1; v.blate = 1; v.adist = 2; show("v4bl+d2", run(&v, n, dummy));
    v = base; v.abytes = big; v.pfa = 1; v.bbytes = big; v.pfb = 1; v.S = 8; v.cS = 1; show("v4tiny S=8, C 16K（活跃集 ~200K）", run(&v, n, dummy));
    v = base; v.abytes = big; v.pfa = 1; v.NT = 2; show("v3t2 只 A 流式, NT=2（A 行/fmopa ×2）", run(&v, n, dummy));
    v = base; v.bbytes = big; v.pfb = 1; v.NT = 2; show("v5t2 只 B 流式, NT=2（B 段换频 ×2）", run(&v, n, dummy));
    v = base; v.abytes = big; v.pfa = 1; v.NT = 8; show("v3t8 只 A 流式, NT=8（A 行/fmopa ÷2）", run(&v, n, dummy));
    v = base; v.abytes = big; v.pfa = 1; v.bbytes = big; v.pfb = 1; v.blate = 1; v.NT = 8; show("v4t8 A、B 流式 NT=8（C 512K）", run(&v, n, dummy));
    v = base; v.abytes = big; v.pfa = 1; v.bbytes = big; v.pfb = 1; v.blate = 1; v.S = 32; v.cS = 32; show("v4s32 A、B 流式 S=32（C 512K）", run(&v, n, dummy));
    v = base; v.abytes = big; v.pfa = 1; v.bbytes = big; v.pfb = 1; v.blate = 1; v.S = 8; v.NT = 8; v.cS = 8; show("v4s8t8 S=8 NT=8（C 256K, B 段 128K）", run(&v, n, dummy));
    // ---- owner 内核（fast_owner 路径）场景：定位 TC1 kloop 11.4 vs TC2 10.1 的来源 ----
    // 访问模式（真实 TC1）：每面板 8 行块任务，任务 i 读 A 块 ch0..ch(i-1)（各 32KB=2 对条带），
    // 每对条带 ×6 个 16 列 t 瓦片重读；B=cbuf（393KB 驻留、行距 768B）；C=cbuf 本行块 16 行瓦片、pend 链回写。
    {
        double* Ap = al(224 * 1024);          // 7 块 × 32KB（HBM，同一面板集重复读）
        double* CB = al(512 * 96 * 8);        // cbuf 393KB
        for (long i = 0; i < 224 * 1024 / 8; i += 5) Ap[i] = 1.0 / (i + 2);
        for (long i = 0; i < 512 * 96; i += 3) CB[i] = 1.0 / (i + 7);
        // o1: 全驻留地板（固定 16KB A 对 + 单 B/C 瓦片）
        {
            uint64_t t0 = cyc();
            for (long r = 0; r < 20000; r++) {
                kloop_owner_bank0(Ap, CB, CB + 8 * 96, dummy, dummy, dummy);
                kloop_owner_bank4(Ap + 1024, CB, CB + 8 * 96 + 16 * 96, dummy + 1024, dummy + 1024, dummy + 128);
            }
            show("o1 owner 对全驻留地板", (double)(cyc() - t0) / 20000 / K * (1.55e9 / (double)cyc_hz()));
        }
        // o2/o5/o6：面板任务模式。o2=真实（A 224K HBM 重读 + cbuf 393K）；o5=A 只 32K（恒 L2 热）；o6=B/C 只用 1 个块（48K 恒热）
        for (int sc = 0; sc < 3; sc++) {
            const long ABLK = sc == 1 ? 1 : 7;      // A 别名到 1 块 → 恒热
            const long BBLK = sc == 2 ? 1 : 8;      // cbuf 只用 1 个 64 行块 → 恒热
            uint64_t t0 = 0; long steps = 0;
            const long panels = 60, warm = 10;
            for (long pnl = 0; pnl < panels; pnl++) {
                if (pnl == warm) t0 = cyc();
                double* pend_c = NULL;
                for (long i = 1; i < 8; i++) {
                    double* blk = CB + (i % BBLK) * 64 * 96;
                    for (long ch = 0; ch < i; ch++) {
                        const double* Bch = CB + (ch % BBLK) * 64 * 96;
                        for (int r = 0; r < 4; r += 2) {
                            const double* A = Ap + ((ch % ABLK) * 4 + r) * 1024;
                            long pfq_off = (((ch % ABLK) * 4 + r + 2) % (4 * ABLK)) * 1024;
                            for (int t = 0; t < 6; t++) {
                                double* cur = blk + 16 * r * 96 + 16 * t;
                                const char* ap = (const char*)(Ap + pfq_off) + t * 2048;
                                const char* ap2 = ap + 6 * 2048;
                                kloop_owner_bank0(A, Bch + 16 * t, pend_c ? pend_c : cur, ap, ap2, (const char*)cur);
                                kloop_owner_bank4(A + 1024, Bch + 16 * t, cur, ap + 1024, ap2 + 1024, (const char*)(cur + 16 * 96));
                                pend_c = cur + 16 * 96;
                                if (pnl >= warm) steps += 1;
                            }
                        }
                    }
                }
            }
            double r = (double)(cyc() - t0) / steps / K * (1.55e9 / (double)cyc_hz());
            show(sc == 0 ? "o2 owner 真实模式（A 224K 重读+cbuf 393K）" : sc == 1 ? "o5 o2 但 A 恒热（32K 别名）" : "o6 o2 但 B/C 恒热（48K 别名）", r);
        }
    }

    sme_stop();
    printf("每 16×32 等效 k 步周期数（理想 8）：\n");
    for (int i = 0; i < nres; i++) printf("%-44s: %6.2f  效率 %.1f%%\n", names[i], vals[i], 800 / vals[i]);
    return 0;
}
