#!/usr/bin/env python3
from pathlib import Path
import json
here=Path(__file__).resolve().parent;out=here/'generated'
def kernel(name,combined):
    lines=['ptrue p0.b','1:']
    for u in range(4):
        lines += [f'ld1b z{reg}.b,p0/z,[%[{ptr}],#{off},mul vl]' for reg,ptr,off in [(0,'a',0),(1,'a',1),(2,'b',0),(3,'b',1)]]
        if combined:
            lines += [f'ld1b z{reg}.b,p0/z,[%[{ptr}],#{off},mul vl]' for reg,ptr,off in [(4,'a2',0),(5,'a2',1),(6,'b2',0),(7,'b2',1)]]
            lines += [f'add z{r}.b,z{r}.b,z{r+4}.b' for r in range(4)]
        lines += [f'smopa za{z}.s,p0/m,p0/m,z{a}.b,z{b}.b' for z,a,b in [(0,0,2),(1,0,3),(2,1,2),(3,1,3)]]
        lines += ['add %[a],%[a],#128','add %[b],%[b],#128']
        if combined:lines += ['add %[a2],%[a2],#128','add %[b2],%[b2],#128']
    lines += ['subs %[K],%[K],#16','b.gt 1b']
    args='const int8_t*a,const int8_t*b,long K'
    ops='[a]"+&r"(a),[b]"+&r"(b),[K]"+&r"(K)'
    if combined:args+=',const int8_t*a2,const int8_t*b2';ops+=',[a2]"+&r"(a2),[b2]"+&r"(b2)'
    return f'static inline __attribute__((always_inline)) void {name}({args}){{\nasm volatile(\n'+''.join(json.dumps(x+'\n\t')+'\n' for x in lines)+f':{ops}::"memory","cc",VCLOB,PCLOB);\n}}\n'
s=kernel('digit_dot',False)+kernel('digit_sumdot',True)
lines=['ptrue p0.b','mov w12,#0','1:']
for z,p,off in [(0,'c',0),(1,'c',1),(2,'cb',0),(3,'cb',1)]:
    lines += [f'mova z{z}.s,p0/m,za{z}h.s[w12,0]',f'st1w z{z}.s,p0,[%[{p}],#{off},mul vl]']
lines += ['add %[c],%[c],#128','add %[cb],%[cb],#128','add w12,w12,#1','cmp w12,#16','b.lt 1b']
s+='static inline __attribute__((always_inline)) void digit_dump(int32_t*c){int32_t*cb=c+512;asm volatile(\n'+''.join(json.dumps(x+'\n\t')+'\n' for x in lines)+':[c]"+&r"(c),[cb]"+&r"(cb)::"memory","cc","x12",VCLOB,PCLOB);}\n'
(out/'digit-kernels.h').write_text(s)
