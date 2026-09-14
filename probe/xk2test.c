// XONE-TC2 内核单元测试：kloop_duplex_bank0/4（NC128）的几何实证 + _x 旁流验证
//   方法：cb1 预填 X 值（模拟装载相）跑基准对；cb2 哨兵 + _x 旁流跑被测对；
//   逐区对比 16 行 × 32 列，谁不一致就暴露真实的排水/填充映射。
#include "../trsm.c"
static double* alz(size_t n){ double* p = aligned_alloc(4096, n*8); memset(p,0,n*8); return p; }
int main(void)
{
    const long ldb = 17088, NCP = 128;
    double* X   = alz((size_t)32*ldb);
    double* cb1 = alz(8192);            // 32 行 × 128 列 = 4096 dbl + 余量
    double* cb2 = alz(8192);
    double* dumm= alz(2048);            // pend==0 排水哑区（ct+cb 跨 8192+... 用 16KB）
    double* dum2= alz(2048);
    double* A   = alz(2048);
    double* Xk  = alz(4096);            // B 操作数 tile 块（2048 dbl bank0 + 1024 bank4 区）
    char* dy    = (char*)alz(512);
    srand(11);
    for (long i = 0; i < 32*ldb; i++) X[i] = (double)(rand()%997)/71.0;
    for (long i = 0; i < 2048; i++)   A[i] = (double)(rand()%997)/89.0;
    for (long i = 0; i < 4096; i++)   Xk[i] = (double)(rand()%997)/79.0;
    for (long r = 0; r < 16; r++)
        for (long j = 0; j < NCP; j++) { cb1[r*NCP+j] = X[r*ldb+j]; cb2[r*NCP+j] = -7777.0; }
    sme_start();
    // 基准对（pend==0：3rd 参数按调用点 (cur)+16）
    sme_zero_za();
    kloop_duplex_bank0(A, Xk, cb1 + 16, dy, dy, dy);
    kloop_duplex_bank4(A, Xk + 1024, cb1, dy+64, dy+64, dy+128);
    // 被测对：pend==0 → bank0 排水目标哑区；旁流 bank0→cur（cols0-15）、bank4→cur+16
    sme_zero_za();
    kloop_duplex_bank0_x(A, Xk, dumm, dy, dy, dy, (const char*)X, (char*)cb2, ldb*8);
    kloop_duplex_bank4_x(A, Xk + 1024, cb2, dy+64, dy+64, dy+128, (const char*)X + 128, (char*)(cb2+16), ldb*8);
    sme_stop();
    (void)dum2;
    long bad=0, badc[2]={0,0};
    for (long r = 0; r < 16; r++)
        for (long j = 0; j < 32; j++) {
            double d = cb1[r*NCP+j] - cb2[r*NCP+j];
            if (d > 1e-9 || d < -1e-9) { bad++; badc[j<16?0:1]++; if (bad<=6) printf("MIS r=%ld j=%ld base=%.4f x=%.4f\n", r, j, cb1[r*NCP+j], cb2[r*NCP+j]); }
        }
    printf("total_bad=%ld cols0-15=%ld cols16-31=%ld %s\n", bad, badc[0], badc[1], bad?"FAIL":"OK");
    return bad!=0;
}
