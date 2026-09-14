// Independent product experiment: streamed residue reconstruction and two-stage packing.
// Not a complete TRSM solver. No official RNG assumptions or cross-call input reuse.
#include "trsm-v55.c"
#include "rns13-fixed-constants.h"
#include "digit-kernels.h"
static size_t packed(int rows,int K,int r,int k) {
    return (size_t)(r/32)*32*K+(size_t)(k/4)*128+(r%32)*4+k%4;
}
static inline __attribute__((always_inline)) void transpose16(const int8_t*src,int8_t*dst,long stride) {
    asm volatile(
      "ptrue p0.s\n\tmov w12,#0\n\t1:\n\t"
      "ld1w {za0h.s[w12,0]},p0/z,[%[src]]\n\tadd %[src],%[src],%[stride]\n\t"
      "add w12,w12,#1\n\tcmp w12,#16\n\tb.lt 1b\n\t"
      "mov w12,#0\n\t2:\n\tst1w {za0v.s[w12,0]},p0,[%[dst]]\n\t"
      "add %[dst],%[dst],#128\n\tadd w12,w12,#1\n\tcmp w12,#16\n\tb.lt 2b"
      :[src]"+&r"(src),[dst]"+&r"(dst):[stride]"r"(stride):"memory","cc","x12",VCLOB,PCLOB);
}
static void encode(const double*src,int8_t*raw,int8_t*dst,int rows,int K,int sequential,double*pack_seconds) {
    const size_t plane=(size_t)rows*K;
    #pragma omp parallel for schedule(static)
    for(int task=0;task<rows/16;task++) {
        svbool_t pg=svptrue_b64();
        for(int r=task*16;r<task*16+16;r++)for(int k=0;k<K;k+=8) {
            svfloat64_t q=svrintn_f64_x(pg,svmul_n_f64_x(pg,svld1_f64(pg,src+(size_t)r*K+k),0x1p45));
            #pragma GCC unroll 13
            for(int p=0;p<RNS_R;p++) {
                svfloat64_t quotient=svrintn_f64_x(pg,svmul_n_f64_x(pg,q,1.0/primes[p]));
                svint64_t v=svcvt_s64_f64_x(pg,svmls_n_f64_x(pg,q,quotient,(double)primes[p]));
                if(sequential)svst1b_s64(pg,raw+(size_t)p*plane+(size_t)r*K+k,v);
                else {
                    int8_t bytes[8];svst1b_s64(pg,bytes,v);
                    size_t pos=(size_t)p*plane+packed(rows,K,r,k);
                    memcpy(dst+pos,bytes,4);memcpy(dst+pos+128,bytes+4,4);
                }
            }
        }
    }
    double t=omp_get_wtime();
    if(sequential) {
        #pragma omp parallel for schedule(static)
        for(int task=0;task<RNS_R*(rows/32);task++) {
            int p=task/(rows/32),r=task%(rows/32)*32;
            sme_start();
            for(int k=0;k<K;k+=64)for(int half=0;half<32;half+=16)
                transpose16(raw+(size_t)p*plane+(size_t)(r+half)*K+k,dst+(size_t)p*plane+packed(rows,K,r+half,k),K);
            sme_stop();
        }
    }
    *pack_seconds+=omp_get_wtime()-t;
}
static inline __attribute__((always_inline)) void residue_add(const int32_t*tmp,uint64_t*acc,int ld,int prime,double invp,uint64_t weight) {
    svbool_t pg=svptrue_b64();
    for(int i=0;i<32;i++)for(int j=0;j<32;j+=8) {
        svfloat64_t v=svcvt_f64_s64_x(pg,svld1sw_s64(pg,tmp+i*32+j));
        svfloat64_t q=svrintn_f64_x(pg,svmul_n_f64_x(pg,v,invp));
        svuint64_t r=svreinterpret_u64_s64(svcvt_s64_f64_x(pg,svmls_n_f64_x(pg,v,q,(double)prime)));
        uint64_t*a=acc+(size_t)i*ld+j;
        svst1_u64(pg,a,svadd_u64_x(pg,svld1_u64(pg,a),svmul_n_u64_x(pg,r,weight)));
    }
}
static void run(const int8_t*A,const int8_t*B,int32_t*partials,double*C,int M,int N,int K,int MB,int NC,int fused) {
    double invp[13];for(int p=0;p<13;p++)invp[p]=1.0/primes[p];
    int rb=M/MB,panels=N/NC;size_t pn=(size_t)M*N;
    #pragma omp parallel
    {
        _Alignas(64) int32_t tmp[1024];
        _Alignas(64) uint64_t acc[256*256];
        if(fused) {
            #pragma omp for schedule(dynamic,1)
            for(int task=0;task<rb*panels;task++) {
                int r0=task%rb*MB,j0=task/rb*NC;
                sme_start();
                svbool_t pg=svptrue_b64();
                for(int z=0;z<MB*NC;z+=8)svst1_u64(pg,acc+z,svdup_u64(0));
                for(int p=0;p<RNS_R;p++)for(int j=0;j<NC;j+=32)for(int i=0;i<MB;i+=32) {
                    sme_zero_za();digit_dot(A+(size_t)p*M*K+(size_t)(r0+i)*K,B+(size_t)p*N*K+(size_t)(j0+j)*K,K);
                    digit_dump(tmp);residue_add(tmp,acc+i*NC+j,NC,primes[p],invp[p],crt_fixed[p]);
                }
                for(int i=0;i<MB;i++)for(int j=0;j<NC;j+=8) {
                    svfloat64_t v=svcvt_f64_s64_x(pg,svreinterpret_s64_u64(svld1_u64(pg,acc+i*NC+j)));
                    svst1_f64(pg,C+(size_t)(r0+i)*N+j0+j,svmul_n_f64_x(pg,v,crt_scale*0x1p-64));
                }
                sme_stop();
            }
        } else {
            #pragma omp for schedule(dynamic,1)
            for(int task=0;task<RNS_R*rb*panels;task++) {
                int r0=task%rb*MB,j0=(task/rb)%panels*NC,p=task/(rb*panels);
                sme_start();svbool_t pg=svptrue_b32();
                for(int j=0;j<NC;j+=32)for(int i=0;i<MB;i+=32) {
                    sme_zero_za();digit_dot(A+(size_t)p*M*K+(size_t)(r0+i)*K,B+(size_t)p*N*K+(size_t)(j0+j)*K,K);
                    digit_dump(tmp);
                    for(int r=0;r<32;r++)for(int c=0;c<32;c+=16)
                        svst1_s32(pg,partials+(size_t)p*pn+(size_t)(r0+i+r)*N+j0+j+c,svld1_s32(pg,tmp+r*32+c));
                }
                sme_stop();
            }
            #pragma omp for schedule(static)
            for(size_t z=0;z<pn;z+=8) {
                sme_start();svbool_t pg=svptrue_b64();svuint64_t value=svdup_u64(0);
                for(int p=0;p<RNS_R;p++) {
                    svfloat64_t v=svcvt_f64_s64_x(pg,svld1sw_s64(pg,partials+(size_t)p*pn+z));
                    svfloat64_t q=svrintn_f64_x(pg,svmul_n_f64_x(pg,v,invp[p]));
                    svuint64_t r=svreinterpret_u64_s64(svcvt_s64_f64_x(pg,svmls_n_f64_x(pg,v,q,(double)primes[p])));
                    value=svadd_u64_x(pg,value,svmul_n_u64_x(pg,r,crt_fixed[p]));
                }
                svst1_f64(pg,C+z,svmul_n_f64_x(pg,svcvt_f64_s64_x(pg,svreinterpret_s64_u64(value)),crt_scale*0x1p-64));
                sme_stop();
            }
        }
    }
}
static uint64_t rng=UINT64_C(0x91db2681ee478c95);
static double value(void) {rng^=rng>>12;rng^=rng<<25;rng^=rng>>27;return ((rng*UINT64_C(2685821657736338717))>>11)*0x1p-53-.5;}
int main(int argc,char**argv) {
    int M=argc>1?atoi(argv[1]):32,N=M,K=argc>2?atoi(argv[2]):512,input=argc>3?atoi(argv[3]):0;
    if(M<32||M%32||K<64||K>2048||K%64||svcntd()!=8)return 2;
    size_t source=(size_t)M*K,plane=(size_t)M*N;
    double*A=aligned_alloc(64,source*8),*B=aligned_alloc(64,source*8),*C=aligned_alloc(64,plane*8);
    int8_t*ad=aligned_alloc(64,source*13),*bd=aligned_alloc(64,source*13),*raw=aligned_alloc(64,source*13),*save=aligned_alloc(64,source*13);
    int32_t*partials=aligned_alloc(64,plane*13*4);
    if(!A||!B||!C||!ad||!bd||!raw||!save||!partials)return 2;
    for(size_t i=0;i<source;i++)A[i]=input==1?.499999999999:input==2?(i%2?.499999999999:-.499999999999):value();
    for(size_t i=0;i<source;i++)B[i]=input==1?.499999999999:input==2?(i%3?.499999999999:-.499999999999):value();
    int count=M==32?1024:128;double ref[1024],quant[1024];
    for(int z=0;z<count;z++) {
        int i=M==32?z/32:(z*73)%M,j=M==32?z%32:(z*121+z/17)%N;long double sum=0;__int128 qsum=0;
        for(int k=0;k<K;k++) {
            double a=A[(size_t)i*K+k],b=B[(size_t)j*K+k];sum+=(long double)a*b;
            qsum+=(__int128)(int64_t)nearbyint(a*0x1p45)*(int64_t)nearbyint(b*0x1p45);
        }
        ref[z]=(double)sum;quant[z]=(double)qsum*0x1p-90;
    }
    double pack=0;encode(A,raw,save,M,K,0,&pack);encode(A,raw,ad,M,K,1,&pack);
    if(memcmp(save,ad,source*13)){puts("ENCODE_EQ FAIL");return 1;}
    encode(B,raw,save,N,K,0,&pack);encode(B,raw,bd,N,K,1,&pack);
    if(memcmp(save,bd,source*13)){puts("ENCODE_EQ FAIL");return 1;}
    // Independent scalar modular check of every encoded coefficient, outside timing.
    for(int r=0;r<M;r++)for(int k=0;k<K;k++)for(int p=0;p<13;p++) {
        int v=(int)((int64_t)nearbyint(A[(size_t)r*K+k]*0x1p45)%primes[p]);
        if(v>primes[p]/2)v-=primes[p];if(v< -primes[p]/2)v+=primes[p];
        if(ad[(size_t)p*source+packed(M,K,r,k)]!=v){puts("ENCODE_SCALAR FAIL");return 1;}
    }
    printf("ENCODE_CHECK coefficients=%zu input=%d PASS\n",source*2,input);
    for(int mode=0;mode<2;mode++)for(int rep=0;rep<3;rep++) {
        pack=0;double t=omp_get_wtime();encode(A,raw,ad,M,K,mode,&pack);encode(B,raw,bd,N,K,mode,&pack);double dt=omp_get_wtime()-t;
        printf("ENCODE_PERF M=%d K=%d sequential=%d rep=%d total_ms=%.6f transpose_ms=%.6f\n",M,K,mode,rep,dt*1e3,pack*1e3);
    }
    for(int MB=32;MB<=256&&MB<=M;MB*=2)for(int NC=32;NC<=256&&NC<=N;NC*=2) {
        if(M!=32&&(MB==32||NC==32))continue;
        for(int fused=0;fused<2;fused++)for(int rep=-1;rep<3;rep++) {
            double t=omp_get_wtime();run(ad,bd,partials,C,M,N,K,MB,NC,fused);double dt=omp_get_wtime()-t;
            double rawerr=0,qerr=0;
            for(int z=0;z<count;z++) {
                int i=M==32?z/32:(z*73)%M,j=M==32?z%32:(z*121+z/17)%N;double got=C[(size_t)i*N+j];
                double e=fabs(got-quant[z]),er=fabs(got-ref[z]);
                if(!isfinite(e)||e>2e-13||!isfinite(er)||er>K*0x1p-46+2e-13){printf("PRODUCT FAIL z=%d got=%.17g quant=%.17g e=%.3e\n",z,got,quant[z],e);return 1;}
                rawerr=fmax(rawerr,er);qerr=fmax(qerr,e);
            }
            printf("FUSED_RNS M=%d N=%d K=%d MB=%d NC=%d fused=%d rep=%d outputs=%d rawerr=%.3e qerr=%.3e ms=%.6f GFLOPS=%.3f PASS\n",M,N,K,MB,NC,fused,rep,count,rawerr,qerr,dt*1e3,2.0*M*N*K/dt*1e-9);
        }
    }
    return 0;
}
