// 配对任务（TRSM_PAIRP）定向正确性：n%128==0 的形状（奇/偶面板数、行块尾、K 分裂），SME vs 通用实现。
// 编译同 gen_test：gcc $FLAGS probe/pair_gen.c trsm.c probe/ref.o -o probe/pair_gen -lm
// 运行时由作业脚本设置 TRSM_JIT_MB=0（强制流水打包）等组合；m>1024 时 direct 默认关，自然进入 special=4。
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

void l_trsm(int m, int n, const double* L, int lda, double* B, int ldb);
void l_trsm_ref(int m, int n, const double* L, int lda, double* B, int ldb);

static double rnd(unsigned* s) { *s = *s * 1103515245u + 12345u; return ((*s >> 8) & 0xffff) / 65536.0 - 0.5; }

static int run_case(int m, int n, int lda, int ldb, unsigned seed)
{
    double* L = malloc((size_t)m * lda * 8);
    double* B = malloc((size_t)m * ldb * 8);
    double* R = malloc((size_t)m * ldb * 8);
    unsigned s = seed;
    for (int i = 0; i < m; i++) {
        double sum = 0;
        for (int k = 0; k < lda; k++) {
            double v = (k < i) ? rnd(&s) : 0.0;
            L[(size_t)i * lda + k] = v;
            if (k < i) sum += fabs(v);
        }
        L[(size_t)i * lda + i] = rnd(&s) + sum + 1.0;
    }
    for (size_t q = 0; q < (size_t)m * ldb; q++) { B[q] = rnd(&s); }
    memcpy(R, B, (size_t)m * ldb * 8);
    l_trsm_ref(m, n, L, lda, R, ldb);
    l_trsm(m, n, L, lda, B, ldb);
    double maxe = 0;
    int bad = 0;
    for (int i = 0; i < m; i++)
        for (int j = 0; j < ldb; j++) {
            double a = B[(size_t)i * ldb + j], b = R[(size_t)i * ldb + j];
            double e = fabs(a - b);
            if (j >= n) { if (a != b) bad++; continue; }   // 越界列不得被改动
            if (isnan(e) || e > 1e-10) bad++;
            if (e > maxe || isnan(e)) maxe = e;
        }
    printf("m=%5d n=%5d lda=%5d ldb=%5d  maxerr=%.2e  %s\n", m, n, lda, ldb, maxe, bad ? "FAIL" : "ok");
    fflush(stdout);
    free(L); free(B); free(R);
    return bad ? 1 : 0;
}

int main(void)
{
    int fails = 0;
    // {m, n, lda, ldb}：npan 偶（配对）/ 奇（单面板 mode4 回退）、m 行块尾（m%192、m%16）、lda/ldb 填充
    int shapes[][4] = {
        {1056, 256, 1056, 256},    // npan=2 配对，m%192=96（半块尾）
        {1056, 512, 1100, 520},    // npan=4 配对 + 填充
        {1216, 384, 1216, 384},    // npan=3 奇 → mode4 单面板
        {1919, 512, 1919, 512},    // m%16≠0（尾条带谓词）
        {2000, 512, 2048, 512},    // lda 填充
        {2432, 768, 2432, 768},    // npan=6 配对，m=TC2 的 m
        {1500, 1024, 1500, 1030},  // npan=8 上限 + ldb 填充
        {3072, 512, 3072, 512},    // 整块 m
        {2304, 256, 2304, 300},    // npan=2 + ldb 填充
        {1152, 128, 1152, 128},    // npan=1 → 单面板
        {192, 256, 192, 256},      // 单行块（i0=0，无 GEMM 段）
        {384, 512, 400, 512},      // 两行块小形状（需 TRSM_DIRECT=0 才走 special）
        {576, 256, 576, 256},
        {2600, 640, 2600, 640},    // npan=5 奇
        {3104, 512, 3104, 512},    // m%192=32（最小尾块）
        {1056, 512, 1056, 513},    // ldb 奇数（行距非 64B 整数倍）
    };
    for (size_t i = 0; i < sizeof(shapes) / sizeof(shapes[0]); i++)
        fails += run_case(shapes[i][0], shapes[i][1], shapes[i][2], shapes[i][3], 4000 + (unsigned)i);
    printf("%s: %d failures\n", fails ? "FAILED" : "ALL PASS", fails);
    return fails ? 1 : 0;
}
