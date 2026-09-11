// 微基准：v33 双 ZA bank 16×16 内核（kloop_duplex_bank0/4）与 16×32 托管内核在 L1 驻留数据上的每 k 步周期数。
// 目的：判断 K 循环 66–75% 的效率损失是内核结构（装载槽/发射）还是访存造成的。
// 编译：gcc -O2 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp probe/kdup.c -o probe/kdup -lm
// 运行：taskset -c 40 ./probe/kdup
#include "../trsm.c"

static double* al(size_t bytes) { void* p = aligned_alloc(4096, bytes); memset(p, 0, bytes); return p; }

int main(void)
{
    const long K = 64;
    double* A = al(K * 16 * 8 * 2);        // 16 行 × K，稠密 128 B/k（两份交替）
    double* B = al(K * 32 * 8 * 2);        // Xp 段：8 个 K×16 微面板（v33 布局），这里只用前两个
    double* C = al(64 * 128 * 8);          // 16 行 × 128 列 C（行距 1 KB），预留两块
    double* scratch = al(8192);
    char* dummy = (char*)al(65536);
    for (long i = 0; i < K * 32; i++) A[i] = 1.0 / (i + 1), B[i] = 1.0 / (i + 3);
    const long reps = 20000;
    sme_start();
    // 1. 双 bank：bank0 + bank4 各 K=64 → 一个 16×32 块的 64 步
    uint64_t t0 = cyc();
    for (long r = 0; r < reps; r++) {
        kloop_duplex_bank0(A, B, C + 16, dummy, dummy, dummy);
        kloop_duplex_bank4(A, B + 1024, C, dummy + 1024, dummy + 1024, dummy + 128);
    }
    uint64_t t1 = cyc();
    double cyc_per_step_dup = (double)(t1 - t0) / reps / K * (1.55e9 / (double)cyc_hz());   // 换成核周期（cntfrq→1.55 GHz）
    // 2. 16×32 托管内核（K=64，全托管回写）：kloop_epi64_sub
    pf_t pf = { dummy, 1, -64, dummy, dummy, 0, NULL };
    uint64_t t2 = cyc();
    for (long r = 0; r < reps; r++) {
        kloop_epi64_sub(A, 128, 128, B, 1024, 32, &pf, scratch, C, 1024, 32);
        za_dump(scratch);
    }
    uint64_t t3 = cyc();
    double cyc_per_step_host = (double)(t3 - t2) / reps / K * (1.55e9 / (double)cyc_hz());
    // 3. 纯 16×32 K 循环（无回写）
    uint64_t t4 = cyc();
    for (long r = 0; r < reps; r++) kloop_plain(A, 128, 128, B, 1024, K, 32, &pf);
    uint64_t t5 = cyc();
    double cyc_per_step_plain = (double)(t5 - t4) / reps / K * (1.55e9 / (double)cyc_hz());
    sme_stop();
    printf("cntfrq=%lu Hz（周期数按 1.55 GHz 核频折算）\n", (unsigned long)cyc_hz());
    printf("16x16 双 bank（bank0+bank4，含回写 C）  : %.2f 周期/(16×32 等效 k 步)  → 效率 %.1f%%（理想 8）\n", cyc_per_step_dup, 800.0 / cyc_per_step_dup);
    printf("16x32 托管回写 kloop_epi64_sub + dump  : %.2f 周期/k 步              → 效率 %.1f%%\n", cyc_per_step_host, 800.0 / cyc_per_step_host);
    printf("16x32 纯 K 循环 kloop_plain（无回写）    : %.2f 周期/k 步              → 效率 %.1f%%\n", cyc_per_step_plain, 800.0 / cyc_per_step_plain);
    return 0;
}
