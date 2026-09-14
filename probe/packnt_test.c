// probe/packnt_test.c —— P0/P2/P1 三合一探针（TC3 pack 相归因 + 候选 B 可行性门，Fable-TC3 设计）
//
// E 系列（P0/P2，pack 引擎单元）：pack_strip_ld4 以 16KB 源（16 行 × 128 k，行距 = TC3 lda）为单位，
// 源/目标各取 热(L2)/冷(DDR) 二态，str-za 与 mova+stnt1d 两种写回 → 分解生产态 ~2630 cyc/16KB 的构成。
// 判据：E1(热源→冷目标, str za) ≤1500 → 写排空不是墙（冷读归因成立，候选 B GO）；
//       E1 ≈2500 → 写墙实锤 → P0 必做且单项 −1ms/线程量级。
// K 系列（P1，kloop 配速）：L2 驻留 duplex chunk 环，K0 bp=哑地址；K1 空槽发真实冷 L 行；
// K2 = K1 + 每 chunk 边界按生产协议（flush pend → 热转置 → zero）。判据：Δ(K1,K0) ≤ +2% → 托管 GO。
//
// 编译：gcc -O2 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp probe/packnt_test.c -o probe/packnt_test -lm
// 运行：numactl -N 1 taskset -c 40 ./probe/packnt_test   ← 不带 --preferred：源/目标落 DDR，与生产 L/pp 同放置
#include "../trsm.c"

static double* al(size_t bytes) { void* p = aligned_alloc(4096, bytes); memset(p, 0, bytes); return p; }

// ---- 几何常量（照 TC3）----
#define LDA 17088L                       // TC3 lda（行距 136,704 B）
#define LDA8 (LDA * 8)
#define WROWS 16L                        // 一个条带窗 = 16 行
#define WBYTES ((size_t)WROWS * LDA8)    // ~2.19 MB / 窗
#define NWIN 96L                         // 冷源窗总数（~210 MB）
#define EKK 128L                         // E 系列每次调用 128 k = 16KB 源 / 16KB 出
#define ECALLS 768L                      // 每变体 768 次（12 窗，每窗 64 次新鲜 k0）

int main(void)
{
    const double c2c = 1.55e9 / (double)cyc_hz();   // 计数器 tick → CPU 周期
    double* src = al(NWIN * WBYTES);                 // 冷 L 源（DDR，首触已置位）
    for (size_t i = 0; i < NWIN * WBYTES / 8; i += 9) src[i] = 1.0 / (double)(i + 3);
    double* dhot = al(64 * 1024);                    // 热目标（16KB 复用 + 余量）
    double* dcold = al(64UL << 20);                  // 冷目标环 64 MB
    char* dummy = (char*)al(1 << 20);
    // K 系列缓冲（全 L2 驻留）
    double* Aring = al(12 * 8192);                   // 12 条 × 8KB
    double* Bpan = al(64 * 1024);                    // Xk 面板 64KB
    double* Cblk = al(192 * 128 * 8);                // C 块 192KB（S=12 × 16 行 × ncp128）
    double* ppnt = al(64UL << 20);                   // K2 的 NT pack 输出环
    for (long i = 0; i < 12 * 1024; i += 5) Aring[i] = 1.0 / (i + 2);
    for (long i = 0; i < 8192; i += 3) Bpan[i] = 1.0 / (i + 7);

    // 逐位一致性：同一 16KB 源，str-za 与 NT 版各打包一次，memcmp
    int bitpass = 0;
    {
        double *o1 = al(16 * 1024), *o2 = al(16 * 1024);
        sme_start();
        pack_strip_ld4(src, LDA, 0, 0, EKK, o1);
        pack_strip_ld4_nt(src, LDA, 0, 0, EKK, o2);
        sme_stop();
        bitpass = memcmp(o1, o2, 16 * 1024) == 0;
    }

    sme_start();

    // ---------------- E 系列：pack 引擎 7 格分解（cyc/16KB）----------------
    // 0=E0 热→热 str | 1=E1 热→冷 str | 2=E2 热→冷 NT | 3=E3 冷→热 str | 4=E4 冷→冷 str（=生产态）
    // 5=E5 冷→冷 NT | 6=E6 载入暖源→冷 NT（托管后 pack 相上界，含暖源载入）
    double ecyc[7];
    for (int v = 0; v < 7; v++) {
        const int coldsrc = (v >= 3 && v <= 5), nt = (v == 2 || v == 5 || v == 6);
        const double* hotwin = src + (NWIN - 1 - v) * (WBYTES / 8);
        if (!coldsrc) for (long r = 0; r < WROWS; r++)
            for (long b = 0; b < EKK * 8; b += 64) { volatile double x = *(const double*)((const char*)(hotwin + r * LDA) + b); (void)x; }
        uint64_t t0 = 0; long done = 0;
        for (long it = 0; it < ECALLS; it++) {
            const double* sp; long k0;
            if (coldsrc) {
                long w = (long)(v - 3) * 12 + (it >> 6);   // 每冷变体独享 12 窗
                k0 = (it & 63) * EKK;
                sp = src + w * (WBYTES / 8);
            } else { sp = hotwin; k0 = 0; }
            double* d = (v == 0 || v == 3) ? dhot : dcold + ((it * 2048) % ((64UL << 20) / 8));
            if (v == 6) {   // E6：pack 前用真实载入暖源
                for (long r = 0; r < WROWS; r++)
                    for (long b = 0; b < EKK * 8; b += 64) { volatile double x = *(const double*)((const char*)(sp + r * LDA + k0) + b); (void)x; }
            }
            if (it == 64) t0 = cyc();
            (nt ? pack_strip_ld4_nt : pack_strip_ld4)(sp, LDA, 0, k0, EKK, d);
            if (it >= 64) done++;
        }
        ecyc[v] = (double)(cyc() - t0) / done * c2c;
    }

    // ---------------- K 系列：duplex 环 空槽配速（cyc/k 步）----------------
    const long KCHUNKS = 256, KWARM = 32;
    double kcyc[3]; double k2pack = 0;
    for (int kv = 0; kv < 3; kv++) {
        uint64_t t0 = 0; long steps = 0; uint64_t tp = 0;
        double* pend_c = NULL;
        long kb = 0, win = 40;                     // 冷区从窗 40 起（避开 E 系列足迹：0-35 + 热窗 89-95）
        size_t ppo = 0;
        for (long chn = 0; chn < KCHUNKS; chn++) {
            if (chn == KWARM) { t0 = cyc(); tp = 0; }
            const double* wbase = src + win * (WBYTES / 8);
            if (kv == 2 && chn > 0) {              // K2：flush pend → 热转置上一 chunk 暖行 → zero
                uint64_t p0t = cyc();
                if (pend_c) { za_flush_duplex_bank4(pend_c + 16, 1024); pend_c = NULL; }
                long pk0 = (kb ? kb : (64 * 2048)) / 8 - 256;
                if (pk0 >= 0) pack_strip_ld4_nt(wbase, LDA, 0, pk0, 256, ppnt + ppo);
                ppo = (ppo + 256 * 16) % ((64UL << 20) / 8);
                sme_zero_za();
                if (chn >= KWARM) tp += cyc() - p0t;
            }
            for (int r = 0; r < 12; r++) {
                const double* A = Aring + ((chn * 12 + r) % 12) * 1024;
                for (int t = 0; t < 4; t++) {
                    double* cur = Cblk + 16L * r * 128 + 32 * t;
                    const char* ap = (const char*)(Aring + ((chn * 12 + r + 1) % 12) * 1024) + t * 2048;
                    const char* bp = dummy;
                    if (kv >= 1 && r < 4) {        // 空槽接线：行 w=4r+t 的 2KB
                        long w = 4 * r + t;
                        bp = (const char*)wbase + w * LDA8 + kb;
                    }
                    const char* pc = (const char*)cur;
                    kloop_duplex_bank0(A, Bpan + t * 2048, (pend_c ? pend_c : cur) + 16, ap, bp, pc);
                    kloop_duplex_bank4(A, Bpan + t * 2048 + 1024, cur, ap + 1024, bp + 1024, pc + 128);
                    pend_c = cur;
                    if (chn >= KWARM) steps += 64;
                }
            }
            kb += 2048;
            if (kb >= 64 * 2048) { kb = 0; win = 40 + ((win - 40 + 1) & 3); }
        }
        if (pend_c) { za_flush_duplex_bank4(pend_c + 16, 1024); pend_c = NULL; }
        kcyc[kv] = ((double)(cyc() - t0) - (kv == 2 ? (double)tp : 0)) / steps * c2c;
        if (kv == 2) k2pack = (double)tp / (KCHUNKS - KWARM) * c2c;
    }

    sme_stop();

    // ---------------- 报告与判据 ----------------
    static const char* en[7] = { "E0 热源→热目标 str-za（引擎地板）", "E1 热源→冷目标 str-za（P2 写墙测试）",
        "E2 热源→冷目标 NT（P0 写侧上界）", "E3 冷源→热目标 str-za（读侧隔离）",
        "E4 冷源→冷目标 str-za（=生产态）", "E5 冷源→冷目标 NT（P0 端到端）",
        "E6 载入暖源→冷 NT（托管后上界，含暖载）" };
    printf("bitwise（str-za vs NT 同源打包）: %s\n", bitpass ? "PASS" : "FAIL");
    printf("== E 系列：pack 引擎（cyc/16KB 源，参考：引擎热速 ~1198 / 生产态 ~2630）==\n");
    for (int v = 0; v < 7; v++) printf("  %-44s: %7.0f\n", en[v], ecyc[v]);
    printf("  [P2 判据] E1=%.0f：%s\n", ecyc[1],
        ecyc[1] <= 1500 ? "≤1500 PASS——写排空不是墙，冷读归因成立，候选 B GO" :
        ecyc[1] >= 2200 ? "≥2200 —— 写墙实锤：P0 必做且单项 −1ms 量级" : "灰区，对照 E2/E5 定夺");
    double dth = (ecyc[4] - ecyc[5]) * 1907.0 / 1.55e9 * 1e3;
    printf("  [P0 定价] E4−E5=%.0f cyc/16KB → 折 TC3 pack 相 ≈ −%.2f ms/线程\n", ecyc[4] - ecyc[5], dth);
    printf("== K 系列：duplex 环 空槽配速（cyc/k 步，理想 8）==\n");
    printf("  K0 哑地址        : %6.2f\n  K1 真实冷 L 配速 : %6.2f（Δ %+0.2f%%）\n  K2 +边界热转置   : %6.2f，插入相 %5.0f cyc/chunk\n",
        kcyc[0], kcyc[1], (kcyc[1] / kcyc[0] - 1) * 100, kcyc[2], k2pack);
    printf("  [P1 判据] Δ(K1,K0)=%+.2f%% → %s\n", (kcyc[1] / kcyc[0] - 1) * 100,
        (kcyc[1] / kcyc[0] - 1) * 100 <= 2.0 ? "PASS——kloop 有余量吸收 pack 行流，托管 GO" : "FAIL——空槽转真流即亏，托管方向关闭");
    printf("  [K2 折算] 插入相 %0.0f cyc/chunk × ~878 chunk/线程 ≈ %.2f ms/线程（须 < E 系列节省量）\n",
        k2pack, k2pack * 878 / 1.55e9 * 1e3);
    return 0;
}
