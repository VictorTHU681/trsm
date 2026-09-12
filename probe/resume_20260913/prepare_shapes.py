#!/usr/bin/env python3
"""Compare equal-area long-K kernels with different operand reuse footprints."""
from pathlib import Path
import json

HERE = Path(__file__).resolve().parent
OUT = HERE/'generated'
base = (OUT/'fullinv-v2.c').read_text()

for mr in (8, 16, 32, 64):
    nr = 512//mr
    ar, bc = mr//8, nr//8
    even = list(range(ar+bc)); odd = list(range(16, 16+ar+bc))
    def loads(reg):
        return [f'ldnt1d z{reg[i]}.d, p0/z, [%[a], #{i}, mul vl]' for i in range(ar)] + [
                f'ld1d z{reg[ar+j]}.d, p0/z, [%[b], #{j}, mul vl]' for j in range(bc)]
    def fmas(reg):
        return [f'fmopa za{i*bc+j}.d, p0/m, p0/m, z{reg[i]}.d, z{reg[ar+j]}.d' for i in range(ar) for j in range(bc)]
    def step(old, new):
        ld, fm, lines = loads(new), fmas(old), []
        for i in range(8):
            for pos in range(i*len(ld)//8, (i+1)*len(ld)//8): lines.append(ld[pos])
            lines.append(fm[i])
        return lines+[f'add %[a], %[a], #{mr*8}', f'add %[b], %[b], #{nr*8}']
    lines=['ptrue p0.d']+loads(even)+[f'add %[a], %[a], #{mr*8}', f'add %[b], %[b], #{nr*8}', '1:']
    for k in range(7): lines += step(even if k%2==0 else odd, odd if k%2==0 else even)
    lines += ['subs %[groups], %[groups], #1', 'b.eq 2f']+step(odd,even)+['b 1b','2:']+fmas(odd)
    kernel='''static inline __attribute__((always_inline)) void inv_shape(const double* a,const double* b,long K) {
    long groups=K/8;
    asm volatile(
'''+''.join(' '+json.dumps(x+'\n\t')+'\n' for x in lines)+'''
    : [a]"+&r"(a),[b]"+&r"(b),[groups]"+&r"(groups) : : "memory","cc",VCLOB,PCLOB);
}
'''
    lines=['ptrue p0.d']
    for row in range(mr):
        lines.append(f'mov w12, #{row%8}')
        for col in range(bc):
            lines += [f'add x9, %[c], #{col*64}', f'st1d {{za{row//8*bc+col}h.d[w12, 0]}}, p0, [x9]']
        lines.append('add %[c], %[c], %[cs]')
    kernel+='''static inline __attribute__((always_inline)) void inv_shape_store(double* c,long cs) {
    asm volatile(
'''+''.join(' '+json.dumps(x+'\n\t')+'\n' for x in lines)+'''
    : [c]"+&r"(c) : [cs]"r"(cs) : "memory","cc","x9","x12",VCLOB,PCLOB);
}
'''
    s=base.replace('#undef l_trsm','#undef l_trsm\n'+kernel,1)
    s=s.replace('(m % 16)',f'(m % {mr})').replace('(n % 32)',f'(n % {nr})')
    s=s.replace(' || n % nc) nc = 32;',f' || nc < {nr} || n % nc) nc = {nr};')
    s=s.replace('strips = m / 16',f'strips = m / {mr}')
    s=s.replace('k < 16 * (r + 1)',f'k < {mr} * (r + 1)').replace('i < 16',f'i < {mr}')
    s=s.replace('r * 16 * m + k * 16 + i',f'r * {mr} * m + k * {mr} + i').replace('(r * 16 + i)',f'(r * {mr} + i)')
    s=s.replace('p < n / 32',f'p < n / {nr}').replace('j < 32',f'j < {nr}')
    s=s.replace('p * m * 32 + k * 32 + j',f'p * m * {nr} + k * {nr} + j').replace('p * 32 + j',f'p * {nr} + j')
    s=s.replace('K = 16 * (r + 1)',f'K = {mr} * (r + 1)').replace('j < nc / 32',f'j < nc / {nr}')
    begin=s.index('                kloop_plain_nt(')
    end=s.index('\n            }',begin)
    s=s[:begin]+f'''                inv_shape(ap + r * {mr} * m, bp + (p * (nc / {nr}) + j) * m * {nr}, K);
                inv_shape_store(B + r * {mr} * ldb + p * nc + j * {nr}, (long)ldb * 8);'''+s[end:]
    s=s.replace('FULLINV m=%d',f'FULLINV mr={mr} nr={nr} m=%d')
    (OUT/f'fullinv-mr{mr}.c').write_text(s)
    print(f'mr={mr} nr={nr} load-vectors-per-K={ar+bc}')
