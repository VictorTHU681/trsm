// TC2 A 驻留波前——构件 1：strip→线程所有权映射与调度表（纯调度，x86 可测）
// 目标：每线程持有一组 16 行条带（A 分片 ≈ 打包 L/nth ≤ ~0.65 MB，跨全部面板驻留 L2），
//       GEMM 工作量（∝ 条带的 K = i0）跨线程配平；每线程调度 = 面板主序、条带升序。
// 不变量自检：全覆盖、无重复、K 配平（max/avg ≤ 1.05）、A 分片 ≤ 预算、块内条带的线程数（diag 收尾扇入）。
//
// 分配算法：条带按权重 w(g) = i0(g) 降序，贪心放入当前累计权重最小的线程（多路配平的 LPT 规则）；
// 权重 0（块 0 只有对角）的条带均匀补位。LPT 对三角权重的 max/avg 界 ≈ 1 + 1/(2·条带数/线程数)。
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { long m, mb, kc, nth, nstrip, nrb; } shape_t;

static int cmp_desc(const void* a, const void* b)
{
    const long* x = a; const long* y = b;
    if (x[1] != y[1]) return x[1] > y[1] ? -1 : 1;
    return x[0] < y[0] ? -1 : 1;
}

int wave_build(shape_t sh, int* owner /*nstrip*/, long* load /*nth*/)
{
    long (*w)[2] = malloc(sizeof(long) * 2 * sh.nstrip);
    for (long g = 0; g < sh.nstrip; g++) {
        long i = (16 * g) / sh.mb;          // 所属行块
        w[g][0] = g; w[g][1] = i * sh.mb;   // 权重 = K = i0
    }
    qsort(w, sh.nstrip, sizeof w[0], cmp_desc);
    memset(load, 0, sizeof(long) * sh.nth);
    for (long q = 0; q < sh.nstrip; q++) {
        int best = 0;
        for (int t = 1; t < sh.nth; t++) if (load[t] < load[best]) best = t;
        owner[w[q][0]] = best;
        load[best] += w[q][1] ? w[q][1] : 1;   // 权重 0 条带记 1，避免全堆一个线程
    }
    free(w);
    return 0;
}

int main(int argc, char** argv)
{
    shape_t sh = { .m = argc > 1 ? atol(argv[1]) : 2432, .mb = 256, .kc = 64, .nth = argc > 2 ? atoi(argv[2]) : 38 };
    sh.nstrip = (sh.m + 15) / 16;
    sh.nrb = (sh.m + sh.mb - 1) / sh.mb;
    int* owner = malloc(sizeof(int) * sh.nstrip);
    long* load = malloc(sizeof(long) * sh.nth);
    wave_build(sh, owner, load);

    // ---- 不变量 ----
    int fail = 0;
    // 1. K 配平
    long mx = 0, sum = 0;
    for (int t = 0; t < sh.nth; t++) { if (load[t] > mx) mx = load[t]; sum += load[t]; }
    double avg = (double)sum / sh.nth, ratio = mx / avg;
    printf("配平: max/avg = %.4f（max %ld avg %.0f）\n", ratio, mx, avg);
    if (ratio > 1.06) { printf("FAIL 配平 >1.06\n"); fail = 1; }
    // 2. A 分片预算（字节 = Σ K×16×8）
    long amax = 0;
    {
        long* ab = calloc(sh.nth, sizeof(long));
        for (long g = 0; g < sh.nstrip; g++) ab[owner[g]] += (16 * g / sh.mb) * sh.mb * 16 * 8;
        for (int t = 0; t < sh.nth; t++) if (ab[t] > amax) amax = ab[t];
        free(ab);
    }
    printf("A 分片最大 %.2f KB（预算 700 KB）\n", amax / 1024.0);
    if (amax > 700 * 1024) { printf("FAIL A 分片超预算\n"); fail = 1; }
    // 3. 每块的所有者扇入（diag 收尾等待的线程数）与块内条带覆盖
    for (long i = 0; i < sh.nrb; i++) {
        long g0 = i * sh.mb / 16, g1 = (i + 1) * sh.mb / 16; if (g1 > sh.nstrip) g1 = sh.nstrip;
        int seen[1024] = {0}, nown = 0;
        for (long g = g0; g < g1; g++) {
            if (owner[g] < 0 || owner[g] >= sh.nth) { printf("FAIL 未覆盖 g=%ld\n", g); fail = 1; }
            if (!seen[owner[g]]) { seen[owner[g]] = 1; nown++; }
        }
        if (i < 4 || i + 1 == sh.nrb) printf("块 %ld: 条带 %ld 个、所有者 %d 个\n", i, g1 - g0, nown);
    }
    // 4. 波前深度示意：每线程最早可开工面板梯次 = 其最小块号（块 i 需等块 <i 的 X）
    long depth = 0;
    for (int t = 0; t < sh.nth; t++) {
        long mn = 1 << 30;
        for (long g = 0; g < sh.nstrip; g++) if (owner[g] == t) { long i = 16 * g / sh.mb; if (i < mn) mn = i; }
        if (mn != (1 << 30) && mn > depth) depth = mn;
    }
    printf("最深线程的最小块号 = %ld（填充梯次上界；nrb=%ld）\n", depth, sh.nrb);
    printf(fail ? "== FAIL ==\n" : "== PASS ==\n");
    return fail;
}
