#!/usr/bin/env python3
from pathlib import Path
HERE=Path(__file__).resolve().parent
OUT=HERE/'generated'
s=(OUT/'strassen-profile.c').read_text()
code='''
    for(int phase=0;phase<2;phase++) {
        if(phase && getenv("DENSE_PRIME"))
            cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,m,n,k,1,a,k,b,n,0,c,n);
#pragma omp parallel
        {
            uint64_t fpcr;asm volatile("mrs %0, fpcr":"=r"(fpcr));
            int tid=omp_get_thread_num();
            if(tid==0 || tid==1 || tid==37)
                fprintf(stderr,"FPCR phase=%d tid=%d fpcr=%016lx\\n",phase,tid,fpcr);
        }
    }
'''
s=s.replace('    int maxdepth=',code+'    int maxdepth=',1)
(OUT/'strassen-prime.c').write_text(s)
