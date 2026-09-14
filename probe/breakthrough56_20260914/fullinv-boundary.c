#include "fullinv-v3.c"

typedef struct { void* mapping; size_t size; double* data; } protected_buffer;
static protected_buffer inv_guard(size_t bytes)
{
    size_t rounded = (bytes + 4095) & ~(size_t)4095;
    void* p = mmap(NULL, rounded + 4096, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED || mprotect((char*)p + rounded, 4096, PROT_NONE)) exit(2);
    return (protected_buffer){p, rounded + 4096, (double*)((char*)p + rounded - bytes)};
}
static uint64_t boundary_state = UINT64_C(0x921e83da725618fc);
static double inv_sample(void)
{
    boundary_state ^= boundary_state >> 12;
    boundary_state ^= boundary_state << 25;
    boundary_state ^= boundary_state >> 27;
    return (double)((boundary_state * UINT64_C(2685821657736338717)) >> 11) * 0x1p-53 - 0.5;
}
int main(void)
{
    long comparisons = 0;
    for (long K = 16; K <= 1024; K += 16) {
        protected_buffer a = inv_guard(K * 128), b = inv_guard(K * 256), c = inv_guard(16 * 96 * 8);
        double ref[16 * 96];
        for (int scale_id = 0; scale_id < 3; scale_id++) {
            double scale = scale_id == 0 ? 0x1p-400 : scale_id == 1 ? 0x1p400 : 1;
            for (long k = 0; k < K * 16; k++) a.data[k] = inv_sample() * scale;
            for (long k = 0; k < K * 32; k++) b.data[k] = inv_sample() / scale;
            for (int q = 0; q < 16 * 96; q++) ref[q] = 37.125;
            for (int i = 0; i < 16; i++) for (int j = 0; j < 32; j++) {
                double v = 0;
                for (long k = 0; k < K; k++) v = fma(a.data[k*16+i], b.data[k*32+j], v);
                ref[i*96+64+j] = v;
            }
            for (int which = 0; which < 2; which++) {
                for (int q = 0; q < 16 * 96; q++) c.data[q] = 37.125;
                sme_start();
                sme_zero_za();
                if (which) inv_compact_pf(a.data, b.data, K);
                else inv_compact(a.data, b.data, K);
                za_store_direct(c.data+64, 96*8, 32);
                sme_stop();
                for (int q = 0; q < 16 * 96; q++) {
                    comparisons++;
                    if (!isfinite(c.data[q]) || c.data[q] != ref[q]) {
                        printf("INVERSE_BOUNDARY FAIL K=%ld scale=%d kernel=%d q=%d actual=%.17g expected=%.17g\n", K, scale_id, which, q, c.data[q], ref[q]);
                        return 1;
                    }
                }
            }
        }
        munmap(a.mapping, a.size); munmap(b.mapping, b.size); munmap(c.mapping, c.size);
    }
    printf("INVERSE_BOUNDARY PASS comparisons=%ld K=16..1024 independent-53bit scaled-inputs protected-A-B-C\n", comparisons);
    return 0;
}
