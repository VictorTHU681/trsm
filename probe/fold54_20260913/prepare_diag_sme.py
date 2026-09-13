#!/usr/bin/env python3
from pathlib import Path
import subprocess,json
here=Path(__file__).resolve().parent
root=here.parents[1]
out=here/'generated'
s=subprocess.check_output(['git','show','36a092c:trsm.c'],cwd=root).decode()
start=s.index('static void diag_prep_sub(')
end=s.index('// ---------------------------------------------------------------- 上下文',start)
reference=s[start:end].replace('diag_prep_sub(', 'diag_prep_reference(',1)
def quoted(lines):return ''.join('    '+json.dumps(line+'\n\t')+'\n' for line in lines)
lines=['ptrue p0.d','zero {za}']
for k in range(16):
    # A is Kx16 and B is Kx16; each quartet computes 16x16.
    lines+=['ld1d z0.d,p0/z,[%[a]]','ld1d z1.d,p0/z,[%[a],#1,mul vl]',
            'ld1d z2.d,p0/z,[%[b]]','ld1d z3.d,p0/z,[%[b],#1,mul vl]',
            'fmopa za0.d,p0/m,p0/m,z0.d,z2.d','fmopa za1.d,p0/m,p0/m,z0.d,z3.d',
            'fmopa za4.d,p0/m,p0/m,z1.d,z2.d','fmopa za5.d,p0/m,p0/m,z1.d,z3.d']
    if k!=15:lines+=['add %[a],%[a],#128','add %[b],%[b],#128']
for tiles in [(0,4),(1,5)]:
    for col in range(8):
        lines += [f'mov w12,#{col}',f'mova z0.d,p0/m,za{tiles[0]}v.d[w12,0]',
                  f'mova z1.d,p0/m,za{tiles[1]}v.d[w12,0]',
                  'fneg z0.d,p0/m,z0.d','fneg z1.d,p0/m,z1.d',
                  'st1d z0.d,p0,[%[d]]','st1d z1.d,p0,[%[d],#1,mul vl]',
                  'add %[d],%[d],#128']
helper='''static int g_diag_sme;
static inline __attribute__((always_inline)) void diag_product_sme(const double* a,const double* b,double* d)
{
    asm volatile(
'''+quoted(lines)+'''
    :[a]"+&r"(a),[b]"+&r"(b),[d]"+&r"(d)::"memory","cc","x12",VCLOB,PCLOB);
}
static __attribute__((noinline)) void diag_transform_sme(const double* L,long lda,long r0,long i0,int s,int vr,const double* inv,double* pk)
{
    double at[256],bt[256];
    for(int k=0;k<16;k++)for(int r=0;r<16;r++)at[k*16+r]=inv[r*16+k];
    sme_start();
    svbool_t pg=svptrue_b64();
    for(int sp=0;sp<s;sp++){
        const double* src=L+r0*lda+i0+16*sp;
        for(int k=0;k<16;k++){
            svfloat64_t lo=svdup_f64(0),hi=lo;
            if(k<vr){lo=svld1_f64(pg,src+k*lda);hi=svld1_f64(pg,src+k*lda+8);}
            svst1_f64(pg,bt+k*16,lo);svst1_f64(pg,bt+k*16+8,hi);
        }
        diag_product_sme(at,bt,pk+256*sp);
    }
    sme_stop();
}
'''
s=s[:start]+helper+s[start:]
old='''    for (int sp = 0; sp < s; sp++) {
        const double* Lb = L + r0 * lda + i0 + 16 * sp;'''
new='''    if(g_diag_sme && s>0)diag_transform_sme(L,lda,r0,i0,s,vr,&Linv[0][0],pk);
    else for (int sp = 0; sp < s; sp++) {
        const double* Lb = L + r0 * lda + i0 + 16 * sp;'''
assert s.count(old)==1
s=s.replace(old,new)
s=s.replace('    c.prof = env_int("TRSM_PROFILE", 0);',
            '    c.prof = env_int("TRSM_PROFILE", 0);\n    g_diag_sme=env_int("TRSM_DIAG_SME",0);',1)
(out/'trsm-diag-sme.c').write_text(s)
(out/'diag-sme-reference.h').write_text(reference)
# The established component test includes extreme reciprocal scales, partial
# rows, both diagonal signs, all output elements and padding guards.
test=(here/'diag_bound_template.c').read_text()
test=test.replace('"trsm-diag-order.c"','"trsm-diag-sme.c"').replace('"diag-reference.h"','"diag-sme-reference.h"')
test=test.replace('g_diag_order','g_diag_sme').replace('mode<4','mode<2').replace('3-pass','1-pass').replace('pass<4','pass<2')
test=test.replace('DIAG_ORDER','DIAG_SME').replace('DIAG_BOUND','DIAG_SME_BOUND')
(out/'diag-sme-check.c').write_text(test)
