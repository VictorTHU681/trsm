#!/usr/bin/env python3
from pathlib import Path
from fractions import Fraction
import json
here=Path(__file__).resolve().parent;out=here/'generated'
primes=[251,241,239,233,229,227,223,211,199,197,193,191,181]
proof=[]
for p in primes:
    magic=((1<<31)+p//2)//p
    limit=2048*(p//2)**2
    bound=Fraction(limit*abs((1<<31)-p*magic),1<<31)+Fraction(p,2)
    assert bound<128
    proof.append(dict(p=p,magic=magic,max_abs_dot=limit,residue_bound=str(bound),bound_float=float(bound)))
(here/'fastmod-bound.json').write_text(json.dumps(proof,indent=2)+'\n')
s=(here/'residue_fused.c').read_text()
start=s.index('static inline __attribute__((always_inline)) void residue_add(')
end=s.index('static void run(',start)
s=s[:start]+r'''
static inline __attribute__((always_inline)) svint32_t fastmod(svint32_t dot,int p) {
    int32_t magic=(int32_t)(((UINT64_C(1)<<31)+p/2)/p);
    svint32_t reciprocal=svdup_s32(magic),quotient;
    asm volatile("sqrdmulh %0.s,%1.s,%2.s":"=w"(quotient):"w"(dot),"w"(reciprocal));
    return svmls_n_s32_x(svptrue_b32(),dot,quotient,p);
}
static inline __attribute__((always_inline)) void residue_add(const int32_t*tmp,uint64_t*acc,int ld,int prime,double invp,uint64_t weight) {
    (void)invp;
    svbool_t pg=svptrue_b64();
    for(int i=0;i<32;i++)for(int j=0;j<32;j+=16) {
        svint32_t r=fastmod(svld1_s32(svptrue_b32(),tmp+i*32+j),prime);
        svuint64_t lo=svreinterpret_u64_s64(svunpklo_s64(r)),hi=svreinterpret_u64_s64(svunpkhi_s64(r));
        uint64_t*a=acc+(size_t)i*ld+j;
        svst1_u64(pg,a,svadd_u64_x(pg,svld1_u64(pg,a),svmul_n_u64_x(pg,lo,weight)));
        svst1_u64(pg,a+8,svadd_u64_x(pg,svld1_u64(pg,a+8),svmul_n_u64_x(pg,hi,weight)));
    }
}
''' + s[end:]
s=s.replace('int32_t*partials,double*C','int8_t*partials,double*C')
old='''                    for(int r=0;r<32;r++)for(int c=0;c<32;c+=16)
                        svst1_s32(pg,partials+(size_t)p*pn+(size_t)(r0+i+r)*N+j0+j+c,svld1_s32(pg,tmp+r*32+c));'''
new='''                    for(int r=0;r<32;r++)for(int c=0;c<32;c+=16)
                        svst1b_s32(pg,partials+(size_t)p*pn+(size_t)(r0+i+r)*N+j0+j+c,fastmod(svld1_s32(pg,tmp+r*32+c),primes[p]));'''
assert old in s;s=s.replace(old,new)
begin=s.index('            #pragma omp for schedule(static)\n            for(size_t z=0;')
end=s.index('\n        }\n    }\n}',begin)
s=s[:begin]+'''            int tid=omp_get_thread_num(),nth=omp_get_num_threads();
            size_t begin=(pn/8*tid/nth)*8,end=(pn/8*(tid+1)/nth)*8;
            sme_start();
            for(size_t z=begin;z<end;z+=8) {
                svbool_t pg=svptrue_b64();svuint64_t value=svdup_u64(0);
                for(int p=0;p<RNS_R;p++) {
                    svuint64_t r=svreinterpret_u64_s64(svld1sb_s64(pg,partials+(size_t)p*pn+z));
                    value=svadd_u64_x(pg,value,svmul_n_u64_x(pg,r,crt_fixed[p]));
                }
                svst1_f64(pg,C+z,svmul_n_f64_x(pg,svcvt_f64_s64_x(pg,svreinterpret_s64_u64(value)),crt_scale*0x1p-64));
            }
            sme_stop();''' + s[end:]
s=s.replace('int32_t*partials=aligned_alloc(64,plane*13*4);','int8_t*partials=aligned_alloc(64,plane*13);')
s=s.replace('for(int r=0;r<M;r++)for(int k=0;k<K;k++)for(int p=0;p<13;p++) {','for(int which=0;which<2;which++)for(int r=0;r<M;r++)for(int k=0;k<K;k++)for(int p=0;p<13;p++) {')
s=s.replace('nearbyint(A[(size_t)r*K+k]*0x1p45)%primes[p]','nearbyint((which?B:A)[(size_t)r*K+k]*0x1p45)%primes[p]')
s=s.replace('if(ad[(size_t)p*source+packed(M,K,r,k)]!=v)', 'if((which?bd:ad)[(size_t)p*source+packed(M,K,r,k)]!=v)')
s=s.replace('FUSED_RNS M=','FASTMOD_RNS M=')
# Explicit vector checks include interval endpoints and independently generated values.
marker='int main(int argc,char**argv) {'
helper='''static int check_fastmod(void) {
    _Alignas(64) int32_t in[16],out[16];int checked=0;
    for(int pidx=0;pidx<13;pidx++) {
        int p=primes[pidx],limit=2048*(p/2)*(p/2);
        for(int rep=0;rep<4096;rep++) {
            for(int j=0;j<16;j++) {
                int z=rep*16+j;
                in[j]=z<256?-limit+z:z<512?limit-(z-256):(int)(value()*(2.0*limit));
            }
            svst1_s32(svptrue_b32(),out,fastmod(svld1_s32(svptrue_b32(),in),p));
            for(int j=0;j<16;j++) {
                if(out[j]< -128||out[j]>127||(in[j]-out[j])%p)return 1;
                checked++;
            }
        }
    }
    printf("FASTMOD_BOUNDARY integers=%d PASS\\n",checked);return 0;
}
'''
assert marker in s;s=s.replace(marker,helper+marker+'\n    if(check_fastmod())return 1;\n    rng=UINT64_C(0x91db2681ee478c95);')
(out/'residue_fastmod.c').write_text(s)
