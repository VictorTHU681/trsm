#!/usr/bin/env python3
from pathlib import Path
here=Path(__file__).resolve().parent
out=here/'generated'
s=(out/'gemm-duplex-layout.c').read_text()
s=s.replace('const long mb=192,nc=128,kc=64;', 'const long mb=env_int("DENSE_MB",192),nc=128,kc=64;\n    if(mb<=0 || mb%16)abort();\n    const int task_order=env_int("DENSE_ORDER",0);')
s=s.replace('long ib=task/np,p=task%np,rows=m-ib*mb,cols=n-p*nc;', 'long ib=task_order?task%blocks:task/np,p=task_order?task/blocks:task%np,rows=m-ib*mb,cols=n-p*nc;')
(out/'gemm-duplex-tasks.c').write_text(s)
