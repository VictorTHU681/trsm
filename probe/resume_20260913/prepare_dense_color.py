#!/usr/bin/env python3
from pathlib import Path
here=Path(__file__).resolve().parent
out=here/'generated'
s=(out/'gemm-duplex-tasks.c').read_text()
s=s.replace('const int task_order=', '''const long ashift=env_int("DENSE_ASHIFT",0),bshift=env_int("DENSE_BSHIFT",0),cshift=env_int("DENSE_CSHIFT",0);
    if(ashift<0||bshift<0||cshift<0||ashift>65536||bshift>65536||cshift>65536||(ashift|bshift|cshift)%64)abort();
    const int task_order=''',1)
s=s.replace('(size_t)blocks*(mb/16)*chunks*1024*8)', '(size_t)blocks*(mb/16)*chunks*1024*8+65536)')
s=s.replace('(size_t)np*chunks*kc*nc*8)', '(size_t)np*chunks*kc*nc*8+65536)')
s=s.replace('(size_t)nth*mb*nc*8)', '(size_t)nth*mb*nc*8+65536)')
beg=s.index('#pragma omp parallel num_threads(nth)',s.index('!dense_reserve'))
s=s[:beg]+'''    double *apdata=dense_a+ashift/8,*bpdata=dense_b+bshift/8,*cpdata=dense_c+cshift/8;
'''+s[beg:].replace('dense_a','apdata').replace('dense_b','bpdata').replace('dense_c','cpdata')
# Avoid renaming dense_copy16 as a side effect of the cpdata replacement.
s=s.replace('cpdataopy16','dense_copy16')
(out/'gemm-duplex-color.c').write_text(s)
