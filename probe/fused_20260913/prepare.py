#!/usr/bin/env python3
from pathlib import Path
import hashlib
import subprocess
here=Path(__file__).resolve().parent
root=here.parents[1]
out=here/'generated'
out.mkdir(exist_ok=True)
s=subprocess.check_output(['git','show','b138512:trsm.c'],cwd=root).decode()
assert hashlib.sha256(s.encode()).hexdigest()=='3bda42d18ac1eb71f29b42a6f80587af807d6109c3684bf9f220f381d7038135'
(out/'trsm-control.c').write_text(s)
start=s.index('#define P4_R(')
end=s.index('// 变体 B：ld2d',start)
base=s[start:end]
parts=[]
for suffix,op,prefix in [('add','fadd','FA4_'),('sub','fsub','FS4_')]:
    q=base.replace('P4_',prefix).replace('pack_strip_ld4','fused_pack_'+suffix)
    q=q.replace('const double* L, long lda, long row0', 'const double* L, long lda, const double* L2, long ldb, long row0')
    lines=q.splitlines(keepends=True)
    new=[]
    for line in lines:
        new.append(line)
        if '"ld4d {' in line:
            new.append('    "ld4d {z8.d, z9.d, z10.d, z11.d}, p0/z, [x10]\\n\\t" \\\n')
            new.append('    "add x10, x10, %[ldb8]\\n\\t" \\\n')
            for i,z in enumerate(('zA','zB','zC','zD')):
                new.append(f'    "{op} " {z} ".d, " {z} ".d, z{8+i}.d\\n\\t" \\\n')
    q=''.join(new)
    q=q.replace('    char* xd =', '    const char* xl2 = (const char*)(L2 + row0 * ldb + k0);\n    const long ldb8 = ldb * 8;\n    char* xd =',1)
    q=q.replace('        "mov x9, %[xl]\\n\\t"', '        "mov x9, %[xl]\\n\\t"\n        "mov x10, %[xl2]\\n\\t"',1)
    q=q.replace('        "add %[xl], %[xl], #256\\n\\t"', '        "add %[xl], %[xl], #256\\n\\t"\n        "add %[xl2], %[xl2], #256\\n\\t"',1)
    q=q.replace(': [xl] "+&r"(xl),', ': [xl] "+&r"(xl), [xl2] "+&r"(xl2),',1)
    q=q.replace(': [lda8] "r"(lda8),', ': [lda8] "r"(lda8), [ldb8] "r"(ldb8),',1)
    q=q.replace('"x9", "x12",', '"x9", "x10", "x12",',1)
    parts.append(q)
(out/'fused-pack.h').write_text('\n'.join(parts))
for name in ('fused.c','fused.h','driver.c','pack_boundary.c','recursive_trsm.c','batch.c'):
    if (here/name).exists():(out/name).write_bytes((here/name).read_bytes())

driver=(here/'driver.c').read_text()
driver=driver.replace('#include "fused.h"', '#include "fused.h"\ndouble fused_compose_ms(void);')
driver=driver.replace('printf("FUSED m=', 'printf("BATCH_FUSED m=')
driver=driver.replace('leaves=%ld max_error=', 'leaves=%ld compose_ms=%.6f max_error=')
driver=driver.replace('st.leaves,worst,', 'st.leaves,fused_compose_ms(),worst,')
(out/'driver-batch.c').write_text(driver)

(out/'bench_trsm.c').write_bytes(subprocess.check_output(['git','show','b138512:bench_trsm.c'],cwd=root))
