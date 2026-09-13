#!/usr/bin/env python3
from pathlib import Path
import json
here=Path(__file__).resolve().parent
out=here/'generated'
generator=(here/'prepare_kernels.py').read_text()
namespace={'json':json}
exec(compile(generator[generator.index('def kernel('):generator.index("definitions='")], 'kernel-generator', 'exec'),namespace)
defs=namespace['kernel']('fold_tile_keep','ld1d',0)+namespace['kernel']('fold_tile_nt','ldnt1d',0)
defs=defs.replace('#768','#256')
lines=['ptrue p0.d']
for r in range(16):
    lines.append(f'mov w12,#{r%8}')
    for j in range(4):lines.append(f'mova z{16+j}.d,p0/m,za{(r//8)*4+j}h.d[w12,0]')
    for j in range(4):lines.append(f'st1d z{16+j}.d,p0,[%[c],#{j},mul vl]')
    lines.append('add %[c],%[c],#256')
defs+='''static inline __attribute__((always_inline)) void fold_tile_store(double* c){
 asm volatile(
'''+''.join(' '+json.dumps(l+'\n\t')+'\n' for l in lines)+'''
 :[c]"+&r"(c)::"memory","cc","x12",VCLOB,PCLOB);
}
static inline svfloat64_t fold_join(svfloat64_t a,svfloat64_t b,int n){
 switch(n){
 case 1:return svext_f64(a,b,1);case 2:return svext_f64(a,b,2);
 case 3:return svext_f64(a,b,3);case 4:return svext_f64(a,b,4);
 case 5:return svext_f64(a,b,5);case 6:return svext_f64(a,b,6);
 case 7:return svext_f64(a,b,7);default:return a;
 }
}
static long run_task_fold_tile(const ctx_t* c,tbuf_t* tb,const task_t* tk,const task_t* next,int tid)
{
 const long i=tk->i,p=tk->p,i0=i*64,m=c->m,ldb=c->ldb,c0=pan_c0(c,p);
 const svbool_t pg=svptrue_b64();
 double* panel=tb->cbuf;
 double* output=c->B+i0*ldb+c0;
 uint64_t start=c->prof>=2?cyc():0,t0=start;
 for(int r=0;r<64;r++)for(int t=0;t<3;t++){
   double* d=panel+(t*m+i0+r)*32;
   const double* b=output+r*ldb+32*t;
   for(int j=0;j<32;j+=8)svst1_f64(pg,d+j,svld1_f64(pg,b+j));
 }
 if(c->prof>=2)tb->t_load+=cyc()-t0;
 t0=c->prof>=2?cyc():0;
 if(c->nobar)while(__atomic_load_n(&c->dcnt[i],__ATOMIC_ACQUIRE)<4)asm volatile("yield");
 if(c->prof>=2)tb->t_wait+=cyc()-t0;
 for(int s=0;s<4;s++){
   long row=i0+16*s,K=row+16;
   const double* a=g_fold_coeff+diag_strip_off((int)(row/16));
   for(int t=0;t<3;t++){
     double* b=panel+t*m*32;
     sme_zero_za();
     if(g_fold_tile==2){PROF_T(c,fold_tile_nt(a,b,K),t_dkloop);}
     else{PROF_T(c,fold_tile_keep(a,b,K),t_dkloop);}
     PROF_T(c,fold_tile_store(b+row*32),t_ddump);
     if(c->prof>=2){tb->n_kstep+=K;tb->n_dkstep+=K;}
   }
 }
 const double* next_b=NULL;const char* next_a=NULL;long next_lines=0;
 if(next && next->kind==0){
   long row=(long)next->i*64;
   if(g_fold_pref&1)next_b=c->B+row*ldb+pan_c0(c,next->p);
   if(g_fold_pref&2){next_a=(const char*)(g_fold_coeff+diag_strip_off((int)(row/16)));next_lines=(row+16)*2;}
 }
 t0=c->prof>=2?cyc():0;
 for(int r=0;r<64;r++){
   double* d=output+r*ldb;
   const double* src=panel+(i0+r)*32;
   if(next_b)for(int line=0;line<12;line++)__builtin_prefetch((const char*)(next_b+r*ldb)+64*line,0,2);
   if(next_a)for(long line=r;line<next_lines;line+=64)__builtin_prefetch(next_a+64*line,0,2);
   long pre=((64-((uintptr_t)d&63))&63)/8,j=0;
   if(pre){svbool_t ph=svwhilelt_b64(0L,pre);svstnt1_f64(ph,d,svld1_f64(ph,src));j=pre;}
   for(;j+8<=96;j+=8){
      long t=j/32,q=j%32;svfloat64_t v;
      if(q<=24)v=svld1_f64(pg,src+t*m*32+q);
      else v=fold_join(svld1_f64(pg,src+t*m*32+24),svld1_f64(pg,src+(t+1)*m*32),(int)(q-24));
      svstnt1_f64(pg,d+j,v);
   }
   if(j<96){svbool_t ph=svwhilelt_b64(j,96L);svstnt1_f64(ph,d+j,svld1_f64(ph,src+(j/32)*m*32+j%32));}
 }
 if(c->prof>=2){tb->t_store+=cyc()-t0;tb->t_task+=cyc()-start;tb->n_task++;}
 __atomic_store_n(&c->flags[i*c->npan+p],1,__ATOMIC_RELEASE);
 (void)tid;
 return -1;
}
'''
s=(out/'trsm-fold-prefetch.c').read_text()
s=s.replace('static int g_fold_owner,g_fold_pref,g_fold_nopack;', 'static int g_fold_owner,g_fold_pref,g_fold_nopack,g_fold_tile;')
s=s.replace('    g_fold_pref=env_int(', '    g_fold_tile=env_int("TRSM_FOLD_TILE",0);\n    g_fold_pref=env_int(')
marker='static inline __attribute__((always_inline)) long run_task_impl('
assert s.count(marker)==1
s=s.replace(marker,defs+'\n'+marker)
s=s.replace('g_fold_owner ? run_task_owner_fold :', 'g_fold_owner ? (g_fold_tile ? run_task_fold_tile : run_task_owner_fold) :')
marker='''                        double* d = buf + (size_t)gr * c.ncp;'''
assert s.count(marker)==1
s=s.replace(marker,'''                        if(g_fold_owner && g_fold_tile){
                            for(int t=0;t<3;t++)for(int j=0;j<32;j+=8)
                                svst1_f64(pg,buf+((long)t*c.m+gr)*32+j,svld1_f64(pg,B+(size_t)gr*ldb+pc0+t*32+j));
                            continue;
                        }
'''+marker)
(out/'trsm-fold-tile.c').write_text(s)
# Direct kernel/ZA-store guard check with the contiguous 32-column layout.
guard=(out/'fold-boundary.c').read_text().replace('trsm-fold-kernels.c','trsm-fold-tile.c')
guard=guard.replace('mode<=5','mode<=3').replace('K*768','K*256').replace('16*768','16*256')
guard=guard.replace('j<96','j<32').replace('k*96','k*32').replace('i*96','i*32').replace('16*96','16*32').replace('j>=64','j>=0')
guard=guard.replace('if(mode==2)fold_nt(a,b+64,K);else if(mode==3)fold_keep(a,b+64,K);else if(mode==4)fold_pf(a,b+64,K);else fold_pfa(a,b+64,K);',
                    'if(mode==2)fold_tile_nt(a,b,K);else fold_tile_keep(a,b,K);')
guard=guard.replace('za_owner_dstore(c+64);','fold_tile_store(c);')
(out/'fold-tile-boundary.c').write_text(guard)
