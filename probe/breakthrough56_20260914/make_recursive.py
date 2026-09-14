#!/usr/bin/env python3
from pathlib import Path
here=Path(__file__).resolve().parent;out=here/'generated'
s=(out/'tiled_right.c').read_text().replace('#include "right-update.h"','#include "right-update.h"\n#include "block_inverse.h"')
s=s.replace('m%16','m%32').replace('nb<16||nb>1024||nb%16','nb<32||nb>1024||(nb&(nb-1))')
begin=s.index('    #pragma omp parallel for schedule(dynamic,1)')
end=s.index('    prep=omp_get_wtime()',begin)
s=s[:begin]+'''    #pragma omp parallel
    {
        size_t bytes=(size_t)nb*nb*8;
        double*iap=aligned_alloc(64,bytes),*ibp=aligned_alloc(64,bytes),*itmp=aligned_alloc(64,bytes);
        if(!iap||!ibp||!itmp)abort(); // This diagnostic prototype does not handle partial allocation failure.
        pin_self(omp_get_thread_num());
        #pragma omp for schedule(dynamic,1)
        for(int block=0;block<(m+nb-1)/nb;block++) {
            int k0=block*nb,K=m-k0<nb?m-k0:nb;
            double*inv=ti+(size_t)block*nb*nb;
            invert_block(L+(size_t)k0*lda+k0,lda,K,inv,nb,iap,ibp,itmp);
            sme_start();
            for(int r=0;r<K;r+=16)pack_strip(inv,nb,K,r,0,r+16,tip+(size_t)block*nb*nb+(size_t)r*nb);
            sme_stop();
        }
        free(itmp);free(ibp);free(iap);
    }
''' + s[end:]
s=s.replace('    double t0=omp_get_wtime()', '    const int jouter=env_int("TRSM_TILE_JOUTER",1);\n    double t0=omp_get_wtime()',1)
old='''                for(int r=rb*mb;r<rend;r+=16)for(int j=0;j<nc;j+=32) {'''
new='''                for(int z=0;z<((rend-rb*mb)/16)*(nc/32);z++) {
                    int r=rb*mb+(jouter?z%((rend-rb*mb)/16):z/(nc/32))*16;
                    int j=(jouter?z/((rend-rb*mb)/16):z%(nc/32))*32;'''
assert old in s;s=s.replace(old,new)
(out/'tiled-recursive.c').write_text(s)
(out/'block_inverse.h').write_text((here/'block_inverse.h').read_text())
