from pathlib import Path
import ast,json,re
p=Path(__file__).resolve().parent
src=(p/'trsm-hoist-ab45.c').read_text()
out='#include "trsm-v49-final.c"\n'
for width in (6,7):
    dec=[x for i in range(8) for x in [*([255]*(8-width)),*range(i*width,(i+1)*width)]]
    enc=[i*8+j for i in range(8) for j in range(8-width,8)]+[255]*(64-8*width)
    for name,data in [('dec',dec),('enc',enc)]:out+=f'static const uint8_t cp{width}{name}[64] __attribute__((aligned(64)))={{'+','.join(map(str,data))+'};\n'

def quote(lines):return ''.join(' '+json.dumps(x+'\n\t')+'\n' for x in lines)

for nc in (128,160,192):
 for width in (8,7,6):
    stride=nc*width
    def setup():
        return ['ptrue p0.d']+([] if width==8 else ['ptrue p1.b',f'mov x11,#{8*width}','whilelo p2.b,xzr,x11',
          'ld1b z28.b,p1/z,[%[dec]]','ld1b z29.b,p1/z,[%[enc]]',f'mov x11,#{1<<(8*(8-width)-1)}','dup z30.d,x11'])
    def loadc(z,ptr,col):
        if width==8:return [f'ld1d z{z}.d,p0/z,[%[{ptr}],#{col},mul vl]']
        addr=f'%[{ptr}]'
        lines=[]
        if col:lines.append(f'add x11,%[{ptr}],#{col*8*width}');addr='x11'
        return lines+[f'ld1b z{z}.b,p2/z,[{addr}]',f'tbl z{z}.b,{{z{z}.b}},z28.b']
    def storec(z,ptr,col):
        if width==8:return [f'st1d z{z}.d,p0,[%[{ptr}],#{col},mul vl]']
        addr=f'%[{ptr}]';lines=[f'add z{z}.d,z{z}.d,z30.d',f'tbl z{z}.b,{{z{z}.b}},z29.b']
        if col:lines.append(f'add x11,%[{ptr}],#{col*8*width}');addr='x11'
        return lines+[f'st1b z{z}.b,p2,[{addr}]']
    extra='' if width==8 else f':[dec]"r"(cp{width}dec),[enc]"r"(cp{width}enc)'
    for bank in (0,4):
        start=src.index(f'static inline __attribute__((always_inline)) void kloop_duplex_bank{bank}(const double* a')
        end=src.index(': [a]',start)
        part=src[start:end].split('asm volatile(',1)[1]
        lines=''.join(ast.literal_eval(x)for x in re.findall(r'"(?:\\.|[^"\\])*"',part)).splitlines()
        code=[]
        for line in lines:
            line=line.strip()
            if line=='ptrue p0.d':code+=setup();continue
            m=re.fullmatch(r'ld1d z(\d+)\.d, p0/z, \[%\[(ct|cb)\], #(\d), mul vl\]',line)
            if m:code+=loadc(int(m[1]),m[2],int(m[3]));continue
            m=re.fullmatch(r'st1d z(\d+)\.d, p0, \[%\[(ct|cb)\], #(\d), mul vl\]',line)
            if m:code+=storec(int(m[1]),m[2],int(m[3]));continue
            m=re.fullmatch(r'add %\[(ct|cb|pc)\], %\[\1\], #1024',line)
            if m:line=f'add %[{m[1]}],%[{m[1]}],#{stride}'
            m=re.fullmatch(r'prfm pldl2keep, \[%\[pc\], #(64|8192|8256)\]',line)
            if m:line=f'prfm pldl2keep,[%[pc],#{ {64:width*8,8192:stride*8,8256:stride*8+width*8}[int(m[1])] }]'
            code.append(line)
        out+=f'''static inline __attribute__((always_inline)) void cp{width}_{nc}_{bank}(const double*a,const double*b,uint8_t*c,const char*ap,const char*bp,const char*pc){{
 char*ct=(char*)c,*cb=ct+{stride*8};asm volatile(
'''+quote(code)+'''
 :[a]"+&r"(a),[b]"+&r"(b),[ct]"+&r"(ct),[cb]"+&r"(cb),[ap]"+&r"(ap),[bp]"+&r"(bp),[pc]"+&r"(pc)
'''+(extra or ':')+''':"memory","cc","x11","x12",VCLOB,PCLOB);}
'''
    code=setup()
    for r in range(8):
        code.append(f'mov w12,#{r}')
        for j in range(4):
            ptr='ct' if j<2 else 'cb';col=j%2
            code.append(f'mova z16.d,p0/m,za{4+j}h.d[w12,0]')
            code+=loadc(17,ptr,col)+['fsub z17.d,z17.d,z16.d']+storec(17,ptr,col)
        code += [f'add %[ct],%[ct],#{stride}',f'add %[cb],%[cb],#{stride}']
    out+=f'''static inline void cp{width}_{nc}_flush(uint8_t*c){{char*ct=(char*)c,*cb=ct+{stride*8};asm volatile(
'''+quote(code)+'''
 :[ct]"+&r"(ct),[cb]"+&r"(cb)
'''+(extra or ':')+''':"memory","cc","x11","x12",VCLOB,PCLOB);}
'''
    out+=f'''static __attribute__((noinline)) double run{width}_{nc}(const double*A,const double*B,uint8_t*C,char*dummy,int mode,int steps,int S){{
 const int NT={nc//32},K=64;size_t na=(mode==2||mode==4)?1024:1,nb=(mode==3||mode==4)?1024:1;size_t pend=0;uint64_t begin=cyc();
 for(int ch=0;ch<steps;ch++)for(int r=0;r<S;r++)for(int j=0;j<NT;j++){{
  size_t aa=((ch%na)*S+r)*1024,an=((ch%na)*S+r+1)*1024;if(r==S-1)an=((ch+1)%na)*S*1024;
  const double*b=B+((ch%nb)*NT+j)*2048;size_t off=(r*16*{nc}+j*32)*{width};uint8_t*c=C+off;
  const char*ap=na>1&&j<4?(char*)(A+an)+j*2048:dummy;const char*bp=dummy;int call=r*NT+j;
  if(nb>1&&call>=(S-8)*NT)bp=(char*)(B+((ch+1)%nb)*NT*2048)+(call-(S-8)*NT)*2048;
  cp{width}_{nc}_0(A+aa,b,C+pend+16*{width},ap,bp,(char*)c);
  cp{width}_{nc}_4(A+aa,b+1024,c,ap+1024,bp+1024,(char*)c+16*{width});pend=off;
 }}
 cp{width}_{nc}_flush(C+pend+16*{width});return(cyc()-begin)/(double)cyc_hz()*1e9/(steps*S*NT*K);
}}
'''

out+=r'''
static double val(uint64_t i){i=(i^(i>>30))*UINT64_C(0xbf58476d1ce4e5b9);i=(i^(i>>27))*UINT64_C(0x94d049bb133111eb);i^=i>>31;return(i>>11)*0x1p-53-.5;}
static double quant(double x,int width){if(width==8)return x;uint64_t u;memcpy(&u,&x,8);int bits=8*(8-width);u=(u+(UINT64_C(1)<<(bits-1)))&~((UINT64_C(1)<<bits)-1);memcpy(&x,&u,8);return x;}
static double unpack(const uint8_t*p,int width){uint64_t u=0;memcpy((char*)&u+8-width,p,width);double x;memcpy(&x,&u,8);return x;}
int main(void){
 size_t ac=(size_t)1024*16*1024,bc=(size_t)1024*6*2048;
 double*A=aligned_alloc(4096,ac*8),*B=aligned_alloc(4096,bc*8);uint8_t*C=aligned_alloc(4096,256*192*8);char*dummy=aligned_alloc(4096,1<<20);
 if(!A||!B||!C||!dummy)return 2;memset(dummy,0,1<<20);
 for(size_t i=0;i<ac;i++)A[i]=val(i+1123);for(size_t i=0;i<bc;i++)B[i]=val(i+3231);
 double(*runs[3][3])(const double*,const double*,uint8_t*,char*,int,int,int)={
 {run8_128,run7_128,run6_128},{run8_160,run7_160,run6_160},{run8_192,run7_192,run6_192}};
 for(int S=12;S<=16;S+=4)for(int n=0;n<3;n++)for(int q=0;q<3;q++){
  int nc=128+n*32,width=8-q,nt=nc/32;memset(C,0,256*192*8);
  sme_start();sme_zero_za();runs[n][q](A,B,C,dummy,4,3,S);sme_stop();int bad=0;double worst=0,rawerr=0;
  for(int i=0;i<S*16;i++)for(int j=0;j<nc;j++){
   double expected=0;long double exact=0;
   for(int ch=0;ch<3;ch++){double sum=0;for(int k=0;k<64;k++){
    double a=A[((size_t)ch*S+i/16)*1024+k*16+i%16],b=B[((size_t)ch*nt+j/32)*2048+(j%32/16)*1024+k*16+j%16];
    sum=fma(a,b,sum);exact-=(long double)a*b;
   }expected=quant(expected-sum,width);}
   double got=unpack(C+((size_t)i*nc+j)*width,width),err=fabs(got-expected);
   worst=fmax(worst,err);rawerr=fmax(rawerr,fabs(got-(double)exact));bad+=!isfinite(err)||err!=0;
  }
  printf("CBUF_CHECK mb=%d nc=%d bytes=%d outputs=%d chunks=3 quant_reference_error=%.3e original_error=%.3e bad=%d %s\n",S*16,nc,width,S*16*nc,worst,rawerr,bad,bad?"FAIL":"PASS");fflush(stdout);if(bad)return 1;
 }
 for(int S=12;S<=16;S+=4)for(int n=0;n<3;n++)for(int mode=1;mode<=4;mode+=3)for(int rep=0;rep<3;rep++)for(int qq=0;qq<3;qq++){
  int q=rep%2?2-qq:qq,width=8-q;memset(C,0,256*192*8);sme_start();sme_zero_za();double ns=runs[n][q](A,B,C,dummy,mode,256,S);sme_stop();
  printf("CBUF_PERF mb=%d nc=%d bytes=%d mode=%d rep=%d ns_per_k=%.6f cycles155=%.6f GFLOPS=%.6f\n",S*16,128+n*32,width,mode,rep,ns,ns*1.55,1024/ns);fflush(stdout);
 }
 free(A);free(B);free(C);free(dummy);return 0;
}
'''
(p/'cbuf-packed50.c').write_text(out)
(p/'404-cbuf-packed50.sh').write_text('''set -euo pipefail
gcc -O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp cbuf-packed50.c -o cbuf-packed50 -lm
OMP_NUM_THREADS=1 numactl -N 1 --preferred=17 taskset -c 40 ./cbuf-packed50
''')
