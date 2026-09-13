#!/usr/bin/env python3
from pathlib import Path
import json
here=Path(__file__).resolve().parent
out=here/'generated'
s=(out/'trsm-diag8.c').read_text()
def quote(lines):return ''.join(' '+json.dumps(x+'\n\t')+'\n' for x in lines)
s=s.replace('static int g_diag8;', '''static int g_diag8;
#ifndef TRSM_DIAG8_CPF
#define TRSM_DIAG8_CPF 0
#endif
#if TRSM_DIAG8_CPF
#define DIAG8_CPF PFC
#else
#define DIAG8_CPF
#endif''')
for bank in (0,4):
    start=s.index(f'static inline __attribute__((always_inline)) void diag8_bank{bank}(')
    end=s.index('\n}',start)+2
    body=s[start:end]
    body=body.replace('    const char* xpf=pf->p;const char* xpf2=pf->p2;',
                      '    const char* xpf=pf->p;const char* xpf2=pf->p2;const char* xpc=pf->pc;')
    body=body.replace('[xpf2]"+&r"(xpf2)', '[xpf2]"+&r"(xpf2),[xpc]"+&r"(xpc)')
    body=body.replace(':[pfn]"r"(pf->pfn)', ':[pcs]"r"(pf->pcs),[pfn]"r"(pf->pfn)')
    fm=[f'fmopa za{bank+j}.d,p0/m,p0/m,z8.d,z{12+j}.d' for j in range(4)]
    ld=['ldnt1d z0.d,p0/z,[%[a]]']+[f'ld1d z{4+j}.d,p0/z,[%[b],#{j},mul vl]' for j in range(4)]
    step=[ld[0],fm[0],ld[1],fm[1],ld[2],fm[2],ld[3],fm[3],ld[4],'add %[a],%[a],#64','add %[b],%[b],#768']
    assert body.count(quote(step))==6
    body=body.replace(quote(step),quote(step)+' DIAG8_CPF\n')
    assert body.count(quote(fm))==2
    body=body.replace(quote(fm),quote(fm)+' DIAG8_CPF\n')
    s=s[:start]+body+s[end:]
start=s.index('static void solve_diag8(');end=s.index('\n}\n',start)+2
body=s[start:end]
body=body.replace('    const double* D=', '    uint64_t begin=c->prof>=2?cyc():0;\n    const double* D=',1)
body=body.replace('        if(ds<7 && t<2)', '''#if TRSM_DIAG8_CPF
        long prefrow=64L*tk->i+8*ds;
        if(prefrow+K/2>c->m)prefrow=c->m-K/2;
        pf.pc=(const char*)(tb->cbuf+prefrow*96+32*t);pf.pcs=768;
#endif
        if(ds<7 && t<2)''')
body=body.replace('PROF_T(c,diag8_bank0(D+diag8_off(ds),cb+32*t,previous,K,&pf),t_dkloop);', 'diag8_bank0(D+diag8_off(ds),cb+32*t,previous,K,&pf);')
body=body.replace('PROF_T(c,diag8_bank4(D+diag8_off(ds),cb+32*t,previous,K,&pf),t_dkloop);', 'diag8_bank4(D+diag8_off(ds),cb+32*t,previous,K,&pf);')
body=body.replace('    PROF_T(c,diag8_flush4(previous),t_ddump);', '    diag8_flush4(previous);\n    if(c->prof>=2)tb->t_dkloop+=cyc()-begin;')
s=s[:start]+body+s[end:]
(out/'trsm-diag8-compact.c').write_text(s)
