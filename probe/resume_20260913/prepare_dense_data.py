#!/usr/bin/env python3
"""Control for sparse input initialization in the historical kdup2 microbenchmark.
Diagnostic inputs are unrelated to the official generator; kernel code is identical.
"""
from pathlib import Path
here=Path(__file__).resolve().parent
out=here/'generated'
s=(out/'micro-control.c').read_text()
helper='''
static double micro_value(uint64_t i) {
    i += UINT64_C(0x9e3779b97f4a7c15);
    i = (i ^ (i >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    i = (i ^ (i >> 27)) * UINT64_C(0x94d049bb133111eb);
    i ^= i >> 31;
    return (double)(i >> 11) * 0x1p-53 - 0.5;
}
'''
s=s.replace('static double* al(',helper+'\nstatic double* al(',1)
start=s.index('    sme_start();',s.index('int main(void)'))
s=s[:start]+'''
    int dense_data=env_int("MICRO_DENSE_DATA",0);
    if(dense_data) {
        double scale=dense_data==1?1.0:0x1p-16;
        for(size_t i=0;i<big/8;i++){Abig[i]=scale*micro_value(i+71);Bbig[i]=micro_value(i+5124511);}
        // Owner aliases B/C and revisits them many times: keep the synthetic
        // triangular factors small enough that this recurrence stays finite.
        for(long i=0;i<224*1024/8;i++)Ap[i]=0x1p-16*micro_value(i+919);
        for(long i=0;i<512*96;i++)CB[i]=micro_value(i+77191);
    }
'''+s[start:]
s=s.replace('    for (int i = 0; i < nres;', '''    for(long i=0;i<512*96;i++)if(!isfinite(CB[i])){printf("MICRO_FINITE FAIL owner\\n");return 2;}
    for(long i=0;i<(8UL<<20)/8;i++)if(!isfinite(Cbig[i])){printf("MICRO_FINITE FAIL dense\\n");return 2;}
    printf("MICRO_FINITE PASS dense_data=%d\\n",dense_data);
    for (int i = 0; i < nres;''',1)
(out/'micro-data.c').write_text(s)
