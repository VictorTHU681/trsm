// 泛化正确性测试：随机 (m, n, lda, ldb) 组合，SME 实现 vs 通用实现（同一 trsm.c 以 -DTRSM_NO_SME_ASM 编译并改名）。
// 编译：gcc -O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp gen_test.c ../trsm.c ref.o -o gen_test -lm
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

void l_trsm(int m, int n, const double* L, int lda, double* B, int ldb);
void l_trsm_ref(int m, int n, const double* L, int lda, double* B, int ldb);

static double rnd(unsigned* s) { *s = *s * 1103515245u + 12345u; return ((*s >> 8) & 0xffff) / 65536.0 - 0.5; }

static int run_case(int m, int n, int lda, int ldb, unsigned seed)
{
    size_t ls=(size_t)m*lda*8,bs=(size_t)m*ldb*8;
    double *lr=malloc(ls+64),*br=malloc(bs+64),*rr=malloc(bs+64);
    double *L=lr+(seed%8),*B=br+((seed/8)%8),*R=rr+((seed/64)%8);
    double *save=malloc(ls);
    if(!lr||!br||!rr||!save)return 1;
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
    memcpy(save,L,ls);
    l_trsm_ref(m, n, L, lda, R, ldb);
    l_trsm(m, n, L, lda, B, ldb);
    double maxe = 0;
    int bad = memcmp(save,L,ls)!=0;
    for (int i = 0; i < m; i++)
        for (int j = 0; j < ldb; j++) {
            double a = B[(size_t)i * ldb + j], b = R[(size_t)i * ldb + j];
            double e = fabs(a - b);
            if (j >= n) { if (a != b) bad++; continue; }   // 越界列不得被改动
            if (isnan(e) || e > 1e-12) bad++;
            if (e > maxe || isnan(e)) maxe = e;
        }
    printf("m=%5d n=%5d lda=%5d ldb=%5d  maxerr=%.2e  %s\n", m, n, lda, ldb, maxe, bad ? "FAIL" : "ok");
    free(lr);free(br);free(rr);free(save);
    return bad ? 1 : 0;
}

int main(int argc, char** argv)
{
    int fails = 0;
    int fixed[][4] = {
        {1, 1, 1, 1}, {1, 40, 5, 41}, {7, 3, 7, 3}, {16, 32, 16, 32}, {17, 33, 20, 40}, {31, 100, 31, 100},
        {33, 7, 64, 9}, {100, 100, 100, 100}, {128, 300, 130, 300}, {255, 257, 256, 260}, {256, 256, 256, 256},
        {257, 33, 300, 64}, {300, 1000, 364, 1064}, {512, 640, 576, 704}, {600, 90, 600, 90}, {777, 555, 800, 600},
        {1030, 260, 1094, 324},
    };
    for (size_t i = 0; i < sizeof(fixed) / sizeof(fixed[0]); i++)
        fails += run_case(fixed[i][0], fixed[i][1], fixed[i][2], fixed[i][3], 1000 + (unsigned)i);
    int nrand = argc > 1 ? atoi(argv[1]) : 40;
    unsigned s = 777;
    for (int t = 0; t < nrand; t++) {
        int m = 1 + (int)((rnd(&s) + 0.5) * 900);
        int n = 1 + (int)((rnd(&s) + 0.5) * 700);
        int lda = m + (int)((rnd(&s) + 0.5) * 70);
        int ldb = n + (int)((rnd(&s) + 0.5) * 70);
        fails += run_case(m, n, lda, ldb, 5000 + t);
    }
    printf("%s: %d failures\n", fails ? "FAILED" : "ALL PASS", fails);
    return fails ? 1 : 0;
}
