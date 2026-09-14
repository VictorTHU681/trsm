// XONE 内核单元测试：kloop_owner_bank0_x/4_x vs bank0/4
//   验证：① 旁流把 X 的 16 行×16 列（行距 ldb*8）逐字节拷到 cbuf 目标（行距 768B）；
//         ② ZA 累加与排水行为与基准内核一致（同输入 → cbuf/C 输出一致）。
// 编译：gcc -O2 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp probe/xktest.c -o probe/xktest -lm
#include "../trsm.c"

static double* alz(size_t n){ double* p = aligned_alloc(4096, n*8); memset(p,0,n*8); return p; }

int main(void)
{
    const long ldb = 20032;               // TC1 行距
    const long NCP = 96;
    double* X   = alz((size_t)64*ldb);    // 64 行源（含垃圾哨兵区）
    double* cb1 = alz(4096*8);            // 基准 cbuf 区（两对 16 行 ×96 列 = 3072 dbl + 余量）
    double* cb2 = alz(4096*8);            // 被测 cbuf 区
    double* A   = alz(2048);              // 一条带对 A（2×1024）
    double* Xk  = alz(64*NCP);            // B 操作数（64 行）
    char*  dummy= (char*)alz(1024);

    srand(7);
    for (long i = 0; i < 64*ldb; i++) X[i] = (double)(rand()%1000)/77.0;
    for (long i = 0; i < 2048; i++)   A[i] = (double)(rand()%1000)/91.0;
    for (long i = 0; i < 64*NCP; i++) Xk[i] = (double)(rand()%1000)/83.0;
    // cbuf 初值：基准区 = X 的 32 行×96 列（模拟装载相已跑）；被测区 = 0xdead 哨兵（旁流应覆盖 16 列子块）
    for (long r = 0; r < 32; r++)
        for (long j = 0; j < NCP; j++) { cb1[r*NCP+j] = X[r*ldb+j]; cb2[r*NCP+j] = -9999.0; }

    sme_start();
    // 基准：t=0 瓦片，pend=0（ct=cur），bank0+bank4
    sme_zero_za();
    kloop_owner_bank0(A, Xk, cb1, dummy, dummy, dummy);
    kloop_owner_bank4(A+1024, Xk, cb1, dummy+64, dummy+64, dummy+128);
    // 被测：同参数 + 旁流（xs=X 对应 16 列子块，xd=cb2 区）
    sme_zero_za();
    kloop_owner_bank0_x(A, Xk, cb2 + 2048 /* pend==0 哑区 */, dummy, dummy, dummy,
                        (const char*)X, (char*)cb2, ldb*8 - 64, 128);
    kloop_owner_bank4_x(A+1024, Xk, cb2, dummy+64, dummy+64, dummy+128,
                        (const char*)(X + 16*ldb), (char*)(cb2 + 16*NCP), ldb*8 - 64, 128);
    sme_stop();

    // 校验 ①：旁流拷贝正确性——cb2 rows0..31 cols0..15 应 == X rows0..31 cols0..15
    long bad = 0;
    for (long r = 16; r < 32 && bad < 8; r++)
        for (long j = 0; j < 16; j++)
            if (cb2[r*NCP+j] != X[r*ldb+j]) {
                if (bad < 8) printf("MISMATCH copy r=%ld j=%ld got=%.6f want=%.6f\n", r, j, cb2[r*NCP+j], X[r*ldb+j]);
                bad++;
            }
    printf(bad ? "COPY FAIL (%ld)\n" : "COPY OK\n", bad);
    // 校验 ②：排水一致性——两区 cols0..15 在 bank4 排水后应一致（基准区被 GEMM 排水修改过同样量）
    long bad2 = 0;
    for (long r = 0; r < 16 && bad2 < 8; r++)
        for (long j = 0; j < 16; j++) {
            double d = cb1[r*NCP+j] - cb2[r*NCP+j];
            if (d > 1e-9 || d < -1e-9) {
                if (bad2 < 8) printf("MISMATCH drain r=%ld j=%ld base=%.6f xone=%.6f\n", r, j, cb1[r*NCP+j], cb2[r*NCP+j]);
                bad2++;
            }
        }
    printf(bad2 ? "DRAIN FAIL (%ld)\n" : "DRAIN OK\n", bad2);
    return bad || bad2;
}
