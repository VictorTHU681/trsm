#!/usr/bin/env python3
"""Move the owner kernel's hosted C update into an earlier K window.

All arithmetic, input loads, output stores and prefetch addresses are retained.
The window=64 reconstruction must have exactly the original instruction order.
"""
from pathlib import Path
from collections import Counter
import ast
import hashlib
import re
import subprocess

here=Path(__file__).resolve().parent
root=here.parents[1]
out=here/'generated'
out.mkdir(exist_ok=True)
def pinned(name):
    return subprocess.check_output(['git','show','b138512:'+name],cwd=root).decode()
source=pinned('trsm.c')
assert hashlib.sha256(source.encode()).hexdigest()=='3bda42d18ac1eb71f29b42a6f80587af807d6109c3684bf9f220f381d7038135'

def instructions(bank,window):
    gap=window//8
    ins=[]
    def emit(s):ins.append(s)
    emit('ptrue p0.d')
    emit('zero {'+', '.join(f'za{bank+i}.d' for i in range(4))+'}')
    emit('mov w12, #0')
    for i in range(4):
        reg=(0,1,4,5)[i];base='a' if i<2 else 'b'
        emit(('ldnt1d' if i<2 else 'ld1d')+f' z{reg}.d, p0/z, [%[{base}]'+(', #1, mul vl' if i%2 else '')+']')
    emit('add %[a], %[a], #128');emit('add %[b], %[b], #768')
    for k in range(64):
        if k<window and k%gap==0:
            emit(f'mov w12, #{k//gap}')
            for q in range(4):
                emit(f'mova z{16+2*q}.d, p0/m, za{4-bank+q}h.d[w12, 0]')
                emit(f'ld1d z{17+2*q}.d, p0/z, [%[{"ct" if q<2 else "cb"}], #{q%2}, mul vl]')
        regs=(0,1,4,5) if k%2==0 else (8,9,12,13)
        nxt=(8,9,12,13) if k%2==0 else (0,1,4,5)
        for q in range(4):
            if k<63:
                emit(('ldnt1d' if q<2 else 'ld1d')+f' z{nxt[q]}.d, p0/z, [%[{"a" if q<2 else "b"}]'+(', #1, mul vl' if q%2 else '')+']')
            emit(f'fmopa za{bank+q}.d, p0/m, p0/m, z{regs[q//2]}.d, z{regs[2+q%2]}.d')
        if k<63:
            emit('add %[a], %[a], #128');emit('add %[b], %[b], #768')
        if k<window:
            for q in range(4):
                if k%gap==((q+1)*gap+3)//4-1:
                    emit(f'fsub z{17+2*q}.d, z{17+2*q}.d, z{16+2*q}.d')
                    emit(f'st1d z{17+2*q}.d, p0, [%[{"ct" if q<2 else "cb"}], #{q%2}, mul vl]')
            if k%gap==gap-1:
                emit('add %[ct], %[ct], #768');emit('add %[cb], %[cb], #768')
        # Keep prefetch timing and addresses exactly as in v50.
        if k%8 in (1,5):
            off=(k//8)*128+(64 if k%8==5 else 0)
            for base in ('ap','bp'):emit(f'prfm pldl2keep, [%[{base}], #{off}]')
        if k%8==3:
            for offset in (0,64,6144,6208):emit(f'prfm pldl2keep, [%[pc], #{(k//8)*768+offset}]')
    return ins

for window in (16,32,48,64):
    s=source
    for bank in (0,4):
        start=s.index(f'static inline __attribute__((always_inline)) void kloop_owner_cursor{bank}(const double* a')
        a=s.index('    asm volatile(',start)+len('    asm volatile(')
        b=s.index('\n: [a]',a)
        original=[ast.literal_eval(x).strip() for x in re.findall(r'"(?:[^"\\]|\\.)*"',s[a:b])]
        original=[x for x in original if x]
        new=instructions(bank,window)
        assert Counter(original)==Counter(new),(bank,window,Counter(original)-Counter(new),Counter(new)-Counter(original))
        if window==64:assert new==original,(bank,'order differs')
        body='\n'+''.join(' "'+x+'\\n\\t"\n' for x in new)
        s=s[:a]+body+s[b:]
    (out/f'trsm-front{window}.c').write_text(s)

# Original guarded test, now calling the current owner cursor kernels explicitly.
boundary=pinned('probe/v50_20260912/v47-boundary.c')
boundary=boundary.replace('trsm-v47-codex.c','trsm-front32.c')
boundary=boundary.replace('kloop_owner_bank0','kloop_owner_cursor0').replace('kloop_owner_bank4','kloop_owner_cursor4')
(out/'front-boundary.c').write_text(boundary)
