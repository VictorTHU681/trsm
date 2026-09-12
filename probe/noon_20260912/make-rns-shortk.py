from pathlib import Path
import json
p=Path(__file__).resolve().parent
ls=['ptrue p0.b','zero {za}','1:']
for u in range(4):
    ls += [f'ld1b z{r}.b,p0/z,[%[{ptr}],#{off},mul vl]' for r,ptr,off in [(0,'a',0),(1,'a',1),(2,'b',0),(3,'b',1)]]
    ls += [f'smopa za{za}.s,p0/m,p0/m,z{a}.b,z{b}.b' for za,a,b in [(0,0,2),(1,0,3),(2,1,2),(3,1,3)]]
    ls += ['add %[a],%[a],#128','add %[b],%[b],#128']
ls += ['subs %[K],%[K],#16','b.gt 1b','mov w12,#0','cbnz %[add],3f','2:']
for za,ptr,off in [(0,'c',0),(1,'c',1),(2,'cb',0),(3,'cb',1)]:
    ls += [f'mova z{za}.s,p0/m,za{za}h.s[w12,0]',f'st1w z{za}.s,p0,[%[{ptr}],#{off},mul vl]']
ls += ['add %[c],%[c],%[cs]','add %[cb],%[cb],%[cs]','add w12,w12,#1','cmp w12,#16','b.lt 2b','b 5f','3:']
for za,ptr,off in [(0,'c',0),(1,'c',1),(2,'cb',0),(3,'cb',1)]:
    ls += [f'ld1w z{za+4}.s,p0/z,[%[{ptr}],#{off},mul vl]',f'mova z{za}.s,p0/m,za{za}h.s[w12,0]',f'add z{za}.s,z{za}.s,z{za+4}.s',f'st1w z{za}.s,p0,[%[{ptr}],#{off},mul vl]']
ls += ['add %[c],%[c],%[cs]','add %[cb],%[cb],%[cs]','add w12,w12,#1','cmp w12,#16','b.lt 3b','5:']
s='''#include "trsm-v47-codex.c"
#include "rns13-fixed-constants.h"
static inline __attribute__((always_inline)) void shortk(const int8_t*a,const int8_t*b,long K,int32_t*c,long cs,long add) {
int32_t*cb=(int32_t*)((char*)c+16*cs);
asm volatile(
'''+''.join(' '+json.dumps(l+'\n\t')+'\n' for l in ls)+''' :[a]"+&r"(a),[b]"+&r"(b),[K]"+&r"(K),[c]"+&r"(c),[cb]"+&r"(cb)
 :[cs]"r"(cs),[add]"r"(add):"memory","cc","x12",VCLOB,PCLOB);
}
'''
s+=r'''
static int64_t qvalue(uint64_t i,uint64_t salt) {
 i+=salt;i=(i^(i>>30))*UINT64_C(0xbf58476d1ce4e5b9);i=(i^(i>>27))*UINT64_C(0x94d049bb133111eb);i^=i>>31;
 return (int64_t)(i&((UINT64_C(1)<<45)-1))-(INT64_C(1)<<44);
}
static size_t packed(int p,int rows,int K,int KC,int r,int k) {
 return (size_t)p*rows*K+(size_t)(k/KC)*rows*KC+(size_t)(r/32)*32*KC+(size_t)((k%KC)/4)*128+(r%32)*4+k%4;
}
static void fill(int8_t*a,int rows,int K,int KC,uint64_t salt) {
 #pragma omp parallel for schedule(static)
 for(int r=0;r<rows;r++)for(int k=0;k<K;k++) {
  int64_t q=qvalue((size_t)r*K+k,salt);
  for(int p=0;p<RNS_R;p++){
   int v=(int)(q%primes[p]);if(v>primes[p]/2)v-=primes[p];if(v<-(primes[p]/2))v+=primes[p];
   a[packed(p,rows,K,KC,r,k)]=v;
  }
 }
}
static void run(const int8_t*A,const int8_t*B,int32_t*C,double*out,int M,int N,int K,int MB,int NC,int KC,int order,int recon) {
 int panels=N/NC,rb=M/MB;size_t pn=(size_t)M*N;
 #pragma omp parallel
 {
  #pragma omp for schedule(dynamic,1)
  for(int task=0;task<RNS_R*panels*rb;task++) {
   int r0=task%rb*MB,panel=task/rb%panels,p=task/(rb*panels),j0=panel*NC;
   int32_t*c=C+(size_t)p*pn+(size_t)r0*N+j0;
   sme_start();
   for(int k=0;k<K;k+=KC) {
    const int8_t *ap=A+(size_t)p*M*K+(size_t)(k/KC)*M*KC+(size_t)r0*KC;
    const int8_t *bp=B+(size_t)p*N*K+(size_t)(k/KC)*N*KC+(size_t)j0*KC;
    if(order==0)for(int j=0;j<NC;j+=32)for(int i=0;i<MB;i+=32)
      shortk(ap+(size_t)i*KC,bp+(size_t)j*KC,KC,c+(size_t)i*N+j,N*4,k);
    else for(int i=0;i<MB;i+=32)for(int j=0;j<NC;j+=32)
      shortk(ap+(size_t)i*KC,bp+(size_t)j*KC,KC,c+(size_t)i*N+j,N*4,k);
   }
   sme_stop();
  }
  if(recon) {
   int tid=omp_get_thread_num(),nt=omp_get_num_threads();
   size_t begin=(pn/16*tid/nt)*16,end=(pn/16*(tid+1)/nt)*16;
   sme_start();
   for(size_t i=begin;i<end;i+=16) {
    svuint64_t acc0=svdup_u64(0),acc1=svdup_u64(0);svbool_t pg=svptrue_b64();
    for(int p=0;p<RNS_R;p++) {
     const int32_t*z=C+(size_t)p*pn+i;
     svfloat64_t f0=svcvt_f64_s64_x(pg,svld1sw_s64(pg,z)),f1=svcvt_f64_s64_x(pg,svld1sw_s64(pg,z+8));
     svfloat64_t q0=svrintn_f64_x(pg,svmul_n_f64_x(pg,f0,1.0/primes[p]));
     svfloat64_t q1=svrintn_f64_x(pg,svmul_n_f64_x(pg,f1,1.0/primes[p]));
     svuint64_t r0=svreinterpret_u64_s64(svcvt_s64_f64_x(pg,svmls_n_f64_x(pg,f0,q0,(double)primes[p])));
     svuint64_t r1=svreinterpret_u64_s64(svcvt_s64_f64_x(pg,svmls_n_f64_x(pg,f1,q1,(double)primes[p])));
     acc0=svadd_u64_x(pg,acc0,svmul_n_u64_x(pg,r0,crt_fixed[p]));acc1=svadd_u64_x(pg,acc1,svmul_n_u64_x(pg,r1,crt_fixed[p]));
    }
    svst1_f64(pg,out+i,svmul_n_f64_x(pg,svcvt_f64_s64_x(pg,svreinterpret_s64_u64(acc0)),crt_scale*0x1p-64));
    svst1_f64(pg,out+i+8,svmul_n_f64_x(pg,svcvt_f64_s64_x(pg,svreinterpret_s64_u64(acc1)),crt_scale*0x1p-64));
   }
   sme_stop();
  }
 }
}
int main(int argc,char**argv) {
 int M=argc>1?atoi(argv[1]):512,N=M,K=2048;size_t pn=(size_t)M*N;int bad=0;
 int8_t*A=aligned_alloc(64,(size_t)RNS_R*M*K),*B=aligned_alloc(64,(size_t)RNS_R*N*K);
 int32_t*C=aligned_alloc(64,(size_t)RNS_R*pn*4);double*out=aligned_alloc(64,pn*8);
 if(!A||!B||!C||!out)return 2;
 /* The small 64x64 check verifies every output against an independent exact sum.
    Larger screening sizes sample 128 pairs; input encoding is outside timing. */
 int count=M==64?M*N:128;
 double*ref=malloc(count*sizeof(double));
 #pragma omp parallel for
 for(int z=0;z<count;z++) {
  int i=M==64?z/N:(z*73)%M,j=M==64?z%N:(z*121+z/17)%N;__int128 sum=0;
  for(int k=0;k<K;k++)sum+=(__int128)qvalue((size_t)i*K+k,UINT64_C(0x8f881))*qvalue((size_t)j*K+k,UINT64_C(0xfe336));
  ref[z]=(double)sum*0x1p-90;
 }
 for(int KC=64;KC<=2048;KC*=2) {
  fill(A,M,K,KC,UINT64_C(0x8f881));fill(B,N,K,KC,UINT64_C(0xfe336));
  for(int MB=64;MB<=128&&MB<=M;MB*=2)for(int NC=64;NC<=128&&NC<=N;NC*=2)for(int order=0;order<1;order++) {
   run(A,B,C,out,M,N,K,MB,NC,KC,order,1);
   double err=0;
   for(int z=0;z<count;z++) {
    int i=M==64?z/N:(z*73)%M,j=M==64?z%N:(z*121+z/17)%N;
    double e=fabs(out[(size_t)i*N+j]-ref[z]);if(!isfinite(e)){bad++;e=INFINITY;}err=fmax(err,e);
   }
   if(!(err<1e-11))bad++;
   printf("SHORTK_CHECK M=%d MB=%d NC=%d KC=%d order=%d outputs=%d qerror=%.3e %s\n",M,MB,NC,KC,order,count,err,err<1e-11?"PASS":"FAIL");fflush(stdout);
   for(int recon=0;recon<=1;recon++)for(int rep=0;rep<3;rep++) {
    double t=omp_get_wtime();for(int z=0;z<3;z++)run(A,B,C,out,M,N,K,MB,NC,KC,order,recon);double dt=(omp_get_wtime()-t)/3;
    printf("SHORTK_PERF M=%d MB=%d NC=%d KC=%d order=%d reconstruct=%d rep=%d ms=%.5f equivalent_GFLOPS=%.3f\n",M,MB,NC,KC,order,recon,rep,dt*1e3,2.0*M*N*K/dt*1e-9);
   }
  }
 }
 free(A);free(B);free(C);free(out);free(ref);return bad?1:0;
}
'''
(p/'rns-shortk.c').write_text(s)
