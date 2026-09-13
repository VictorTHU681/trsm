#!/usr/bin/env python3
from pathlib import Path
import subprocess,json
here=Path(__file__).resolve().parent
root=here.parents[1]
out=here/'generated'
s=subprocess.check_output(['git','show','36a092c:trsm.c'],cwd=root).decode()
def quote(lines):return ''.join(' '+json.dumps(x+'\n\t')+'\n' for x in lines)
def kernel(bank):
    other=4-bank
    even=(0,4,5,6,7);odd=(8,12,13,14,15)
    def loads(reg):return [f'ldnt1d z{reg[0]}.d,p0/z,[%[a]]']+[f'ld1d z{reg[j+1]}.d,p0/z,[%[b],#{j},mul vl]' for j in range(4)]
    def fmas(reg):return [f'fmopa za{bank+j}.d,p0/m,p0/m,z{reg[0]}.d,z{reg[j+1]}.d' for j in range(4)]
    def step(old,new):
        ld,fm=loads(new),fmas(old)
        return [ld[0],fm[0],ld[1],fm[1],ld[2],fm[2],ld[3],fm[3],ld[4],'add %[a],%[a],#64','add %[b],%[b],#768']
    def drain(row):return [f'mov w12,#{row}']+[f'mova z{16+j}.d,p0/m,za{other+j}h.d[w12,0]' for j in range(4)]+[f'st1d z{16+j}.d,p0,[%[cp],#{j},mul vl]' for j in range(4)]+['add %[cp],%[cp],#768']
    code=f'''static inline __attribute__((always_inline)) void diag8_bank{bank}(const double* a,const double* b,double* cp,long K,const pf_t* pf)
{{
    const char* xpf=pf->p;const char* xpf2=pf->p2;
    long pfc=pf->pfn,remaining=K-8;
    asm volatile(
'''+quote(['ptrue p0.d','zero {'+', '.join(f'za{bank+j}.d' for j in range(4))+'}']+loads(even)+['add %[a],%[a],#64','add %[b],%[b],#768'])
    for k in range(7):
        code+=quote(step(even if k%2==0 else odd,odd if k%2==0 else even)+drain(k))
        if k%2==0:code+=' PFL PFX\n'
    code+=quote(fmas(odd)+drain(7)+['cbz %[remaining],2f','1:']+loads(even)+['add %[a],%[a],#64','add %[b],%[b],#768'])
    for k in range(7):
        code+=quote(step(even if k%2==0 else odd,odd if k%2==0 else even))
        if k%2==0:code+=' PFL PFX\n'
    code+=quote(fmas(odd)+['subs %[remaining],%[remaining],#8','b.ne 1b','2:'])
    code+=''' :[a]"+&r"(a),[b]"+&r"(b),[cp]"+&r"(cp),[remaining]"+&r"(remaining),[xpf]"+&r"(xpf),[pfc]"+&r"(pfc),[xpf2]"+&r"(xpf2)
 :[pfn]"r"(pf->pfn),[pfj]"r"(pf->pfj):"memory","cc","x12",VCLOB,PCLOB);
}
'''
    return code
helper='static int g_diag8;\n'+kernel(0)+kernel(4)
lines=['ptrue p0.d']
for r in range(8):lines += [f'mov w12,#{r}']+[f'mova z{16+j}.d,p0/m,za{4+j}h.d[w12,0]' for j in range(4)]+[f'st1d z{16+j}.d,p0,[%[cp],#{j},mul vl]' for j in range(4)]+['add %[cp],%[cp],#768']
helper+='''static inline void diag8_flush4(double* cp){asm volatile(
'''+quote(lines)+''' :[cp]"+&r"(cp)::"memory","cc","x12",VCLOB,PCLOB);}
static inline size_t diag8_off(int s){return (size_t)s*(s+1)*32;}
static void diag8_prep_pair(const double* L,long lda,long i0,int pair,double* block)
{
    for(int ss=2*pair;ss<2*pair+2;ss++){
        long row=i0+8*ss;
        double inv[8][8]={{0}},local[8][8];
        for(int r=0;r<8;r++)for(int c=0;c<8;c++)local[r][c]=c<=r?L[(row+r)*lda+row+c]:0;
        for(int c=0;c<8;c++){
            inv[c][c]=1.0/local[c][c];
            for(int r=c+1;r<8;r++){
                double v=0;
                for(int q=c;q<r;q++)v+=local[r][q]*inv[q][c];
                inv[r][c]=-v/local[r][r];
            }
        }
        double* dst=block+diag8_off(ss);
        for(int sp=0;sp<ss;sp++){
            double lower[8][8],product[8][8]={{0}};
            for(int t=0;t<8;t++)for(int q=0;q<8;q++)lower[t][q]=L[(row+t)*lda+i0+8*sp+q];
            for(int r=0;r<8;r++)for(int t=0;t<=r;t++)for(int q=0;q<8;q++)product[r][q]+=inv[r][t]*lower[t][q];
            for(int q=0;q<8;q++)for(int r=0;r<8;r++)dst[64*sp+q*8+r]=-product[r][q];
        }
        for(int q=0;q<8;q++)for(int r=0;r<8;r++)dst[64*ss+q*8+r]=inv[r][q];
    }
}
'''
marker='static void diag_prep_sub('
assert s.count(marker)==1
s=s.replace(marker,helper+'\n'+marker)
marker='static inline __attribute__((always_inline)) long run_task_impl('
solver='''static void solve_diag8(const ctx_t* c,tbuf_t* tb,const task_t* tk,const task_t* next,double* cb)
{
    const double* D=c->diag+c->diag_off[tk->i];
    const double* nb=NULL;long nrows=0,done=0,nlines=12;
    if(next && next->kind==0 && next->nseg==1){
        nb=c->B+(long)next->i*64*c->ldb+pan_c0(c,next->p);nrows=64;
        nlines=(pan_nci(c,next->p)*8+63)/64;
    }
    double* previous=tb->scratch2;
    sme_zero_za();
    for(int ds=0;ds<8;ds++)for(int t=0;t<3;t++){
        long K=8*(ds+1);
        pf_t pf={(const char*)tb->scratch2,1,-64,(const char*)tb->scratch2,(const char*)tb->scratch2,0,NULL};
        if(ds<7 && t<2)pf.p2=(const char*)(D+diag8_off(ds+1))+(size_t)t*(K/2)*64;
        if(nb && done<nrows){
            pf.p=(const char*)(nb+done*c->ldb);pf.pfn=nlines;pf.pfj=c->ldb*8-nlines*64;
            long rows=(K/2)/nlines;if(rows<1)rows=1;done+=rows;
        }
        if((3*ds+t)%2==0){PROF_T(c,diag8_bank0(D+diag8_off(ds),cb+32*t,previous,K,&pf),t_dkloop);}
        else{PROF_T(c,diag8_bank4(D+diag8_off(ds),cb+32*t,previous,K,&pf),t_dkloop);}
        previous=cb+(8L*ds)*96+32*t;
        if(c->prof>=2){tb->n_kstep+=K/2;tb->n_dkstep+=K/2;}
    }
    PROF_T(c,diag8_flush4(previous),t_ddump);
}

'''
assert s.count(marker)==1
s=s.replace(marker,solver+marker)
start=s.index(marker);end=s.index('static long run_task(',start)
body=s[start:end]
needle='''    {
        const double* D = c->diag + c->diag_off[i];'''
assert body.count(needle)==1
body=body.replace(needle,'''    if(fast_owner && g_diag8)solve_diag8(c,tb,tk,tkn,cb);
    else {
        const double* D = c->diag + c->diag_off[i];''')
s=s[:start]+body+s[end:]
marker='    g_implicit_owner = env_int('
s=s.replace(marker,'''    g_diag8=env_int("TRSM_DIAG8",0) && c.special==5 && m<=1024 && c.ddirect && c.cshift==0;
    if(env_int("TRSM_EXPECT_DIAG8",-1)>=0 && g_diag8!=env_int("TRSM_EXPECT_DIAG8",-1))abort();
'''+marker,1)
old='''                diag_prep_sub(L, lda, i * mb, (int)mbi, ditem_s[q], c.diag + c.diag_off[i] + diag_strip_off(ditem_s[q]));'''
new='''                if(g_diag8)diag8_prep_pair(L,lda,i*mb,ditem_s[q],c.diag+c.diag_off[i]);
                else diag_prep_sub(L, lda, i * mb, (int)mbi, ditem_s[q], c.diag + c.diag_off[i] + diag_strip_off(ditem_s[q]));'''
assert s.count(old)==1
s=s.replace(old,new)
(out/'trsm-diag8.c').write_text(s)
driver=(out/'independent-fold.c').read_text().replace('setenv("TRSM_FOLD_OWNER",reuse?"0":"1",1);','setenv("TRSM_DIAG8",reuse?"0":"1",1);').replace('setenv("TRSM_EXPECT_FOLD",reuse?"0":"1",1);','setenv("TRSM_EXPECT_DIAG8",reuse?"0":"1",1);')
(out/'independent-diag8.c').write_text(driver)
