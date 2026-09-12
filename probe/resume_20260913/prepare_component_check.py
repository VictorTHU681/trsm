#!/usr/bin/env python3
"""Broaden component checks without changing the already-recorded pilot source."""
from pathlib import Path
here=Path(__file__).resolve().parent
out=here/'generated'
s=(out/'strassen-profile.c').read_text()
s=s.replace('    long checks=full?(long)m*n:128;', '''    const int edge[]={0,1,7,8,15,16,31,32,63,64,95,96,127,128,191,192,255,256};
    int ii[32],jj[32],ni=0,nj=0;
    for(int t=0;t<18;t++){if(edge[t]<m)ii[ni++]=edge[t];if(edge[t]<n)jj[nj++]=edge[t];}
    const int tails[]={1,8,16,32,64};
    for(int t=0;t<5;t++){if(tails[t]<=m)ii[ni++]=m-tails[t];if(tails[t]<=n)jj[nj++]=n-tails[t];}
    ii[ni++]=m/2;ii[ni++]=m/2-1;jj[nj++]=n/2;jj[nj++]=n/2-1;
    long checks=full?(long)m*n:128+ni*nj;''')
s=s.replace('        long double ref=0,norm=0;', '''        if(!full && q>=128){i=ii[(q-128)/nj];j=jj[(q-128)%nj];}
        long double ref=0,norm=0;''')
s=s.replace('        leaf_seconds=sum_seconds=output_seconds=0;', '''        // Same allocation, fresh dense input before every timed product.
#pragma omp parallel for schedule(static)
        for(long i=0;i<(long)m*k;i++)a[i]=value(i+23879+UINT64_C(104729)*(rep+1)+depth*967);
#pragma omp parallel for schedule(static)
        for(long i=0;i<(long)k*n;i++)b[i]=value(i+UINT64_C(127421711)+UINT64_C(15485863)*(rep+1)+depth*1549);
        leaf_seconds=sum_seconds=output_seconds=0;''')
s=s.replace('m<=128&&n<=128', 'm<=384&&n<=384')
s=s.replace('"128-samples"', '"samples-plus-edges-fresh-input"')
(out/'strassen-check.c').write_text(s)
