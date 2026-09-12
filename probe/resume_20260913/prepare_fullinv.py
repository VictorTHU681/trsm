#!/usr/bin/env python3
"""Capacity-only workspace reuse and private bookkeeping for full-inverse prototype."""
from pathlib import Path
import json
import re

HERE = Path(__file__).resolve().parent
OUT = HERE/'generated'
source = (HERE/'fullinv.c').read_text()
(OUT/'fullinv.c').write_text(source)
helper = '''
// Capacity is reusable; no input-dependent values are reused by the algorithm.
static double *inv_space, *ap_space, *bp_space, *dummy_space;
static size_t inv_capacity, ap_capacity, bp_capacity, dummy_capacity;
static int reserve_full(double** p, size_t* capacity, size_t need) {
    if (*p && *capacity >= need) return 1;
    if (*p) trsm_free_buffer(*p, *capacity);
    *p = NULL; *capacity = 0;
    const size_t alignment = (size_t)2 << 20;
    size_t sz = (need + alignment - 1) & ~(alignment - 1);
    void* raw = mmap(NULL, sz + alignment, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (raw == MAP_FAILED) return 0;
    uintptr_t v = ((uintptr_t)raw + alignment - 1) & ~(uintptr_t)(alignment - 1);
    size_t lead = v - (uintptr_t)raw;
    if (lead) munmap(raw, lead);
    if (alignment > lead) munmap((void*)(v + sz), alignment - lead);
    madvise((void*)v, sz, MADV_HUGEPAGE);
    *p = (double*)v; *capacity = sz;
    return 1;
}
'''
source = source.replace('#undef l_trsm', '#undef l_trsm\n'+helper, 1)
begin = source.index('    double* inv = aligned_alloc')
end = source.index('#pragma omp parallel for', begin)
source = source[:begin] + '''    if (!reserve_full(&inv_space, &inv_capacity, ibytes) ||
        !reserve_full(&ap_space, &ap_capacity, ibytes) ||
        !reserve_full(&bp_space, &bp_capacity, bbytes) ||
        !reserve_full(&dummy_space, &dummy_capacity, 262144)) {
        baseline_trsm(m, n, L, lda, B, ldb); return;
    }
    double *inv = inv_space, *ap = ap_space, *bp = bp_space;
    char* dummy = (char*)dummy_space;
    memset(dummy, 0, 262144);
    const int private_sw = env_int("TRSM_FULLINV_PRIVATE", 1);
''' + source[end:]
source = source.replace('        pf_t pf = { dummy, PF_CONTIG_LINES, 0, dummy, dummy, 0, NULL };',
'''        sw_t private_state = {0};
        pf_t pf = { dummy, PF_CONTIG_LINES, 0, dummy, dummy, 0,
                    private_sw ? &private_state : NULL };''', 1)
source = source.replace('    free(dummy); free(bp); free(ap); free(inv);',
'''    // inv is reset to identity and solved again next call; ap/bp are completely
    // repacked over every range the product will read. Keep only allocation capacity.''', 1)
(OUT/'fullinv-v2.c').write_text(source)

def compact(name, prefetch):
    lines = ['ptrue p0.d']
    def loads(regs):
        return [f'ldnt1d z{regs[0]}.d, p0/z, [%[a]]',
                f'ldnt1d z{regs[1]}.d, p0/z, [%[a], #1, mul vl]'] + [
                f'ld1d z{regs[2+j]}.d, p0/z, [%[b], #{j}, mul vl]' for j in range(4)]
    def fmas(regs):
        return [f'fmopa za{r*4+c}.d, p0/m, p0/m, z{regs[r]}.d, z{regs[c+2]}.d'
                for r in range(2) for c in range(4)]
    even, odd = (0, 1, 4, 5, 6, 7), (8, 9, 12, 13, 14, 15)
    def step(old, new):
        ld, fm = loads(new), fmas(old)
        seq = [ld[0], fm[0], fm[1], ld[1], fm[2], fm[3],
               ld[2], fm[4], ld[3], fm[5], ld[4], fm[6], ld[5], fm[7]]
        if prefetch: seq += ['prfm pldl2keep, [%[a], #1024]', 'prfm pldl2keep, [%[b], #2048]']
        return seq + ['add %[a], %[a], #128', 'add %[b], %[b], #256']
    lines += loads(even) + ['add %[a], %[a], #128', 'add %[b], %[b], #256', '1:']
    for k in range(7): lines += step(even if k%2==0 else odd, odd if k%2==0 else even)
    lines += ['subs %[groups], %[groups], #1', 'b.eq 2f']
    lines += step(odd, even) + ['b 1b', '2:'] + fmas(odd)
    return f'''static inline __attribute__((always_inline)) void {name}(const double* a, const double* b, long K) {{
    long groups = K / 8;
    asm volatile(
''' + ''.join(' '+json.dumps(line+'\n\t')+'\n' for line in lines) + '''
        : [a]"+&r"(a), [b]"+&r"(b), [groups]"+&r"(groups)
        : : "memory", "cc", VCLOB, PCLOB);
}
'''

header = compact('inv_compact', False) + compact('inv_compact_pf', True)
header += '''
static inline __attribute__((always_inline)) void inv_copy32(const double* src, double* dst) {
    asm volatile(
        "ptrue p0.d\\n\\t"
        "ld1d z0.d, p0/z, [%[src]]\\n\\t"
        "ld1d z1.d, p0/z, [%[src], #1, mul vl]\\n\\t"
        "ld1d z2.d, p0/z, [%[src], #2, mul vl]\\n\\t"
        "ld1d z3.d, p0/z, [%[src], #3, mul vl]\\n\\t"
        "stnt1d z0.d, p0, [%[dst]]\\n\\t"
        "stnt1d z1.d, p0, [%[dst], #1, mul vl]\\n\\t"
        "stnt1d z2.d, p0, [%[dst], #2, mul vl]\\n\\t"
        "stnt1d z3.d, p0, [%[dst], #3, mul vl]\\n\\t"
        : : [src]"r"(src), [dst]"r"(dst) : "memory", "v0", "v1", "v2", "v3", "p0");
}
'''
(OUT/'fullinv-compact.h').write_text(header)
source = source.replace('#undef l_trsm', '#undef l_trsm\n#include "fullinv-compact.h"', 1)
source = source.replace('    const int private_sw =', '''    const int compact_kernel = env_int("TRSM_FULLINV_KERNEL", 1);
    const int packed_copy = env_int("TRSM_FULLINV_PACK", 1);
    const int private_sw =''', 1)
old = '''        for (long p = 0; p < n / 32; p++) {
            for (long k = 0; k < m; k++)
                for (long j = 0; j < 32; j++)
                    bp[p * m * 32 + k * 32 + j] = B[k * ldb + p * 32 + j];
        }'''
new = '''        for (long p = 0; p < n / nc; p++) {
            if (packed_copy) {
                sme_start();
                for (long k = 0; k < m; k++)
                    for (long j = 0; j < nc / 32; j++)
                        inv_copy32(B + k * ldb + p * nc + j * 32,
                                   bp + (p * (nc/32) + j) * m * 32 + k * 32);
                sme_stop();
            } else {
                for (long j = 0; j < nc / 32; j++)
                    for (long k = 0; k < m; k++)
                        for (long v = 0; v < 32; v++)
                            bp[(p * (nc/32) + j) * m * 32 + k * 32 + v] = B[k * ldb + p * nc + j * 32 + v];
            }
        }'''
assert old in source
source = source.replace(old, new, 1)
old = '''                kloop_plain_nt(ap + r * 16 * m, 128, 128,
                               bp + (p * (nc / 32) + j) * m * 32, 256, K, 32, &pf);'''
new = '''                if (compact_kernel == 1)
                    inv_compact(ap + r * 16 * m, bp + (p * (nc / 32) + j) * m * 32, K);
                else if (compact_kernel == 2)
                    inv_compact_pf(ap + r * 16 * m, bp + (p * (nc / 32) + j) * m * 32, K);
                else
                    kloop_plain_nt(ap + r * 16 * m, 128, 128,
                                   bp + (p * (nc / 32) + j) * m * 32, 256, K, 32, &pf);'''
assert old in source
source = source.replace(old, new, 1)
(OUT/'fullinv-v3.c').write_text(source)
audit = source.replace('#undef l_trsm', '#undef l_trsm\nstatic long inv_completed;\nlong get_inv_completed(void) { return inv_completed; }', 1)
audit = audit.replace('    double t5 = omp_get_wtime();', '    double t5 = omp_get_wtime();\n    ++inv_completed;', 1)
(OUT/'fullinv-v3-audit.c').write_text(audit)
test = (HERE.parents[1]/'probe/v50_20260912/independent-cursor-switch.c').read_text()
test = test.replace('int main(void)', 'long get_inv_completed(void);\nint main(void)', 1)
test = re.sub(r'            setenv\("TRSM_(?:OWNER|OWNER_CURSOR|EXPECT_CURSOR|EXPECT_OWNER)"[^\n]+\n', '', test)
test = test.replace('            l_trsm(m,n,L,lda,B,ldb);', '''            setenv("TRSM_FULLINV_KERNEL", (mode&1)?"2":"1", 1);
            setenv("TRSM_FULLINV_PACK", (mode&1)?"1":"0", 1);
            l_trsm(m,n,L,lda,B,ldb);
            if (get_inv_completed() != cases+1) abort();''', 1)
(OUT/'independent-fullinv.c').write_text(test)
(OUT/'fullinv-boundary.c').write_bytes((HERE/'fullinv-boundary.c').read_bytes())
