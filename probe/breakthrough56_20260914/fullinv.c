// Research prototype only: recompute inverse and pack every input on every call.
// Include the pinned solver as the inversion implementation and fallback.
#define l_trsm baseline_trsm
#include "trsm-control.c"
#undef l_trsm

static long inv_prefix(long t, long strips)
{
    long panel = t / strips, r = t % strips;
    return panel * strips * (strips + 1) / 2 + r * (r + 1) / 2;
}

static long inv_bound(long target, long tasks, long strips)
{
    long lo = 0, hi = tasks;
    while (lo < hi) {
        long mid = lo + (hi - lo) / 2;
        if (inv_prefix(mid, strips) < target) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

void l_trsm(int m, int n, const double* L, int lda, double* B, int ldb)
{
    if (!env_int("TRSM_FULLINV", 1) || m <= 0 || m > 4096 || (m % 16) || n <= 0 || (n % 32) || !sme_f64_available()) {
        baseline_trsm(m, n, L, lda, B, ldb);
        return;
    }
    const int prof = env_int("TRSM_FULLINV_PROFILE", 0);
    int nc = env_int("TRSM_FULLINV_NC", 128);
    if ((nc != 32 && nc != 64 && nc != 128) || n % nc) nc = 32;
    const int nth = omp_get_max_threads();
    const long strips = m / 16, panels = n / nc, tasks = strips * panels;
    const size_t ibytes = (size_t)m * m * 8, bbytes = (size_t)m * n * 8;
    double t0 = omp_get_wtime();
    double* inv = aligned_alloc(4096, (ibytes + 4095) & ~(size_t)4095);
    double* ap = aligned_alloc(4096, (ibytes + 4095) & ~(size_t)4095);
    double* bp = aligned_alloc(4096, (bbytes + 4095) & ~(size_t)4095);
    char* dummy = aligned_alloc(4096, 262144);
    if (!inv || !ap || !bp || !dummy) {
        free(inv); free(ap); free(bp); free(dummy);
        baseline_trsm(m, n, L, lda, B, ldb);
        return;
    }
    memset(dummy, 0, 262144);
    madvise(inv, ibytes, MADV_HUGEPAGE);
    madvise(ap, ibytes, MADV_HUGEPAGE);
    madvise(bp, bbytes, MADV_HUGEPAGE);
#pragma omp parallel for schedule(static) num_threads(nth)
    for (long i = 0; i < m; i++) {
        memset(inv + i * m, 0, (size_t)m * 8);
        inv[i * m + i] = 1.0;
    }
    double t1 = omp_get_wtime();
    baseline_trsm(m, m, L, lda, inv, m);
    double t2 = omp_get_wtime();
    // Save all RHS entries before writing any output, preserving in-place semantics.
#pragma omp parallel num_threads(nth)
    {
        pin_self(omp_get_thread_num());
#pragma omp for schedule(static)
        for (long r = 0; r < strips; r++) {
            for (long k = 0; k < 16 * (r + 1); k++)
                for (long i = 0; i < 16; i++)
                    ap[r * 16 * m + k * 16 + i] = inv[(r * 16 + i) * m + k];
        }
#pragma omp for schedule(static)
        for (long p = 0; p < n / 32; p++) {
            for (long k = 0; k < m; k++)
                for (long j = 0; j < 32; j++)
                    bp[p * m * 32 + k * 32 + j] = B[k * ldb + p * 32 + j];
        }
    }
    double t3 = omp_get_wtime();
#pragma omp parallel num_threads(nth)
    {
        const int tid = omp_get_thread_num(), nt = omp_get_num_threads();
        pin_self(tid);
        long total = inv_prefix(tasks, strips);
        long lo = inv_bound(total * tid / nt, tasks, strips);
        long hi = inv_bound(total * (tid + 1) / nt, tasks, strips);
        pf_t pf = { dummy, PF_CONTIG_LINES, 0, dummy, dummy, 0, NULL };
        sme_start();
        for (long task = lo; task < hi; task++) {
            long p = task / strips, r = task % strips;
            long K = 16 * (r + 1);
            for (long j = 0; j < nc / 32; j++) {
                sme_zero_za();
                kloop_plain_nt(ap + r * 16 * m, 128, 128,
                               bp + (p * (nc / 32) + j) * m * 32, 256, K, 32, &pf);
                za_store_direct(B + r * 16 * ldb + p * nc + j * 32, (long)ldb * 8, 32);
            }
        }
        sme_stop();
    }
    double t4 = omp_get_wtime();
    free(dummy); free(bp); free(ap); free(inv);
    double t5 = omp_get_wtime();
    if (prof) fprintf(stderr, "FULLINV m=%d n=%d nc=%d init_ms=%.6f inverse_ms=%.6f pack_ms=%.6f gemm_ms=%.6f free_ms=%.6f total_ms=%.6f\n",
                      m, n, nc, 1e3*(t1-t0), 1e3*(t2-t1), 1e3*(t3-t2), 1e3*(t4-t3), 1e3*(t5-t4), 1e3*(t5-t0));
}
