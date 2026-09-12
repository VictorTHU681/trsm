#!/usr/bin/env python3
"""Generate isolated KC64 scheduling experiments from the pinned production tree."""
from pathlib import Path
import ast
import hashlib
import json
import re
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REV = '2fcdc7a'
SOURCE = subprocess.check_output(['git', 'show', f'{REV}:trsm.c'], cwd=ROOT).decode()
assert hashlib.sha256(SOURCE.encode()).hexdigest() == '3bda42d18ac1eb71f29b42a6f80587af807d6109c3684bf9f220f381d7038135'
OUT = HERE / 'generated'
OUT.mkdir(exist_ok=True)


def get_function(source, bank):
    start = source.index(f'static inline __attribute__((always_inline)) void kloop_owner_cursor{bank}(const double* a')
    end = source.index('PCLOB);}', start) + len('PCLOB);}')
    return start, end, source[start:end]


def kernel(bank, mode):
    other = 4 - bank
    a = ['ptrue p0.d', 'zero {' + ', '.join(f'za{x}.d' for x in range(bank, bank+4)) + '}']
    a += ['ldnt1d z0.d, p0/z, [%[a]]', 'ldnt1d z1.d, p0/z, [%[a], #1, mul vl]',
          'ld1d z4.d, p0/z, [%[b]]', 'ld1d z5.d, p0/z, [%[b], #1, mul vl]',
          'add %[a], %[a], #128', 'add %[b], %[b], #768']
    pre = [[] for _ in range(64)]
    post = [[] for _ in range(64)]
    # Each bank drains exactly 32 vectors from the opposite ZA bank.
    window = 4 if mode == 'front32' else 8
    offset = 32 if mode == 'late32' else 0
    if mode == 'late32':
        window = 4
    for row in range(8):
        start = offset + row * window
        pre[start].append(f'mov w12, #{row}')
        for pair in range(4):
            ptr, col = ('ct' if pair < 2 else 'cb'), pair % 2
            reg = 16 + 2 * pair
            load_step = start + (pair * 2 if mode == 'spread' else 0)
            pre[load_step] += [f'mova z{reg}.d, p0/m, za{other+pair}h.d[w12, 0]',
                               f'ld1d z{reg+1}.d, p0/z, [%[{ptr}], #{col}, mul vl]']
            store_step = start + (pair + 1) * window // 4 - 1
            post[store_step] += [f'fsub z{reg+1}.d, z{reg+1}.d, z{reg}.d',
                                 f'st1d z{reg+1}.d, p0, [%[{ptr}], #{col}, mul vl]']
        post[start + window - 1] += ['add %[ct], %[ct], #768', 'add %[cb], %[cb], #768']
    for k in range(64):
        a += pre[k]
        old = (0, 1, 4, 5) if k % 2 == 0 else (8, 9, 12, 13)
        nxt = (8, 9, 12, 13) if k % 2 == 0 else (0, 1, 4, 5)
        row, phase = divmod(k, 8)
        pf_ptr, pf_off = [('ap', row*128), ('bp', row*128),
                          ('pc', row*768), ('pc', row*768+64),
                          ('ap', row*128+64), ('bp', row*128+64),
                          ('pc', row*768+6144), ('pc', row*768+6208)][phase]
        for t, (ar, br) in enumerate([(old[0], old[2]), (old[0], old[3]), (old[1], old[2]), (old[1], old[3])]):
            if k < 63:
                op, ptr, lane = ('ldnt1d', 'a', t) if t < 2 else ('ld1d', 'b', t-2)
                a.append(f'{op} z{nxt[t]}.d, p0/z, [%[{ptr}], #{lane}, mul vl]')
            # A single prefetch per K step, dispersed through the FMOPA stream.
            if t == 2:
                a.append(f'prfm pldl2keep, [%[{pf_ptr}], #{pf_off}]')
            a.append(f'fmopa za{bank+t}.d, p0/m, p0/m, z{ar}.d, z{br}.d')
        if k < 63:
            a += ['add %[a], %[a], #128', 'add %[b], %[b], #768']
        a += post[k]
    assert sum(x.startswith('fmopa') for x in a) == 256
    assert sum(x.startswith('mova') for x in a) == 32
    assert sum(x.startswith('st1d') for x in a) == 32
    assert sum(x.startswith('prfm') for x in a) == 64
    _, _, oldfun = get_function(SOURCE, bank)
    begin = oldfun.index('asm volatile(') + len('asm volatile(')
    end = oldfun.index(': [a]')
    oldasm = ''.join(ast.literal_eval(x) for x in re.findall(r'"(?:\\.|[^"\\])*"', oldfun[begin:end]))
    oldpf = sorted(x.strip() for x in oldasm.splitlines() if x.strip().startswith('prfm'))
    assert oldpf == sorted(x for x in a if x.startswith('prfm'))
    return oldfun[:begin] + '\n' + ''.join(' ' + json.dumps(x+'\n\t') + '\n' for x in a) + oldfun[end:]


for mode in ('control', 'pfspread', 'spread', 'front32', 'late32'):
    source = SOURCE
    if mode != 'control':
        for bank in (0, 4):
            start, end, _ = get_function(source, bank)
            source = source[:start] + kernel(bank, mode) + source[end:]
    (OUT/f'trsm-{mode}.c').write_text(source)
    boundary = (ROOT/'probe/v50_20260912/v49-final-boundary.c').read_text()
    boundary = boundary.replace('trsm-v49-final.c', f'trsm-{mode}.c')
    (OUT/f'boundary-{mode}.c').write_text(boundary)
    micro = (ROOT/'probe/kdup2.c').read_text()
    micro = micro.replace('../trsm.c', f'trsm-{mode}.c')
    micro = micro.replace('kloop_owner_bank0(', 'kloop_owner_cursor0(').replace('kloop_owner_bank4(', 'kloop_owner_cursor4(')
    (OUT/f'micro-{mode}.c').write_text(micro)
    print(mode, hashlib.sha256(source.encode()).hexdigest())
(OUT/'bench_trsm.c').write_bytes((ROOT/'bench_trsm.c').read_bytes())
