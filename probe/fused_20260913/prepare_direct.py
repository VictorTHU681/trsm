#!/usr/bin/env python3
"""Assign all seven products of a logical output tile to one worker.

Input packing remains batched. Direct C updates avoid the seven M buffers and
the final composition pass. Ownership prevents overlapping output writers.
"""
from pathlib import Path
here=Path(__file__).resolve().parent
out=here/'generated'
s=(here/'batch.c').read_text()
def replace(old,new):
    global s
    assert s.count(old)==1,(old,s.count(old))
    s=s.replace(old,new)
replace('!reserve(&all_m,&all_mcap,7*msize*8)||','')
replace('for(long task=0;task<7*ntasks;task++){','for(long task=0;task<ntasks;task++){')
replace('long product=interleave?task%7:task/ntasks,local=interleave?task/7:task%ntasks;', 'long local=task;')
replace('            const double *apack=all_a+product*asize,*bpack=all_b+product*bsize;',
'''            double *c11=c+ib*mb*ldc+p*nc,*c12=c11+nh,*c21=c11+mh*ldc,*c22=c21+nh;
            for(long product=0;product<7;product++){
            const double *apack=all_a+product*asize,*bpack=all_b+product*bsize;''')
replace('            emit(cb,all_m+product*msize+local*mb*nc,rows,cols,nc,1,1);',
'''            switch(product){
                case 0: emit(cb,c11,rows,cols,ldc,alpha,!beta);emit(cb,c22,rows,cols,ldc,alpha,!beta);break;
                case 1: emit(cb,c21,rows,cols,ldc,alpha,!beta);emit(cb,c22,rows,cols,ldc,-alpha,0);break;
                case 2: emit(cb,c12,rows,cols,ldc,alpha,!beta);emit(cb,c22,rows,cols,ldc,alpha,0);break;
                case 3: emit(cb,c11,rows,cols,ldc,alpha,0);emit(cb,c21,rows,cols,ldc,alpha,0);break;
                case 4: emit(cb,c11,rows,cols,ldc,-alpha,0);emit(cb,c12,rows,cols,ldc,alpha,0);break;
                case 5: emit(cb,c22,rows,cols,ldc,alpha,0);break;
                case 6: emit(cb,c11,rows,cols,ldc,alpha,0);break;
            }''')
a=s.index('            sme_stop();\n        }\n#pragma omp master\n        gemm_end=')
b=s.index('    double end=omp_get_wtime();',a)
s=s[:a]+'''            sme_stop();
            }
        }
    }
'''+s[b:]
replace('    compose_ms=1e3*(end-gemm_end);','    compose_ms=0;')
s=s.replace('// Batch all seven products within this call. Source quadrants are read once for',
'''// Each worker owns a logical output tile across all seven products.
// Pack all seven products within this call. Source quadrants are read once for''')
(out/'direct.c').write_text(s)
(out/'driver-direct.c').write_text((out/'driver-batch.c').read_text().replace('BATCH_FUSED','DIRECT_FUSED'))
