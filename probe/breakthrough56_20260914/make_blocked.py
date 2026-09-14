#!/usr/bin/env python3
"""Blocked inverse application; each call still rebuilds the complete inverse."""
from pathlib import Path
here=Path(__file__).resolve().parent
out=here/'generated'
s=(out/'fullinv-v3.c').read_text()
s=s.replace('    const int nth = omp_get_max_threads();','''    const int nth = omp_get_max_threads();
    int mb=env_int("TRSM_INV_MB",128), kc=env_int("TRSM_INV_KC",64);
    if(mb<16 || mb>256 || mb%16)mb=128;
    if(kc<16 || kc>256 || kc%16)kc=64;
    const int blocked=env_int("TRSM_INV_BLOCKED",1);
    const int nrb=(m+mb-1)/mb;''')
begin=s.index('    // Save all RHS entries')
end=s.index('    double t3 =',begin)
s=s[:begin]+'''    // Read every RHS value before the in-place product overwrites any output.
#pragma omp parallel num_threads(nth)
    {
        pin_self(omp_get_thread_num());
#pragma omp for schedule(dynamic,1)
        for(long r=0;r<strips;r++) {
            sme_start();
            pack_strip(inv,m,m,r*16,0,16*(r+1),ap+r*16*m);
            sme_stop();
        }
        // Short row ranges allow neighboring panels to reuse source cache lines.
#pragma omp for schedule(static)
        for(long task=0;task<(long)nrb*(n/nc);task++) {
            long rb=task/(n/nc),p=task%(n/nc),end=(rb+1)*mb;
            if(end>m)end=m;
            sme_start();
            for(long k=rb*mb;k<end;k++)for(long j=0;j<nc/32;j++)
                inv_copy32(B+k*ldb+p*nc+j*32,bp+(p*(nc/32)+j)*m*32+k*32);
            sme_stop();
        }
    }
''' + s[end:]
begin=s.index('#pragma omp parallel num_threads(nth)',s.index('    double t3 ='))
end=s.index('    double t4 =',begin)
old=s[begin:end]
s=s[:begin]+'''    if(!blocked) {
''' + old + '''    } else {
#pragma omp parallel num_threads(nth)
        {
            _Alignas(64) double acc[256*128];
            pin_self(omp_get_thread_num());
#pragma omp for schedule(dynamic,1)
            for(long task=0;task<(long)nrb*(n/nc);task++) {
                long rb=task%(long)nrb,p=task/nrb;
                long r0=rb*mb/16,r1=(rb+1)*mb/16;
                if(r1>strips)r1=strips;
                sme_start();
                for(long k=0;k<r1*16;k+=kc) {
                    for(long r=r0;r<r1;r++) {
                        long K=(r+1)*16-k;
                        if(K<=0)continue;
                        if(K>kc)K=kc;
                        for(long j=0;j<nc/32;j++) {
                            double*dst=acc+(r-r0)*16*nc+j*32;
                            if(!k)sme_zero_za();
                            else za_load_direct(dst,nc*8,32);
                            inv_compact(ap+r*16*m+k*16,bp+(p*(nc/32)+j)*m*32+k*32,K);
                            za_store_direct(dst,nc*8,32);
                        }
                    }
                }
                for(long r=0;r<(r1-r0)*16;r++)for(long j=0;j<nc;j+=32)
                    inv_copy32(acc+r*nc+j,B+(r0*16+r)*ldb+p*nc+j);
                sme_stop();
            }
        }
    }
''' + s[end:]
(out/'fullinv-blocked.c').write_text(s)
