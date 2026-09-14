#!/usr/bin/env python3
"""Preload only the current thread's owned row interval of each RHS panel."""
from pathlib import Path
here=Path(__file__).resolve().parent;out=here/'generated'
s=(out/'trsm-v56.c').read_text()
old='    int nexta;              // fast_owner'
assert s.count(old)==1
s=s.replace(old,'    int preload58;          // experimental per-call RHS loading order\n'+old)
old='    if (c.special == 5) c.loadpf = env_int("TRSM_LOADPF", 0);'
s=s.replace(old,old+'\n    c.preload58=env_int("TRSM_PRELOAD58",0) && c.special==5 && !c.cshift;')
begin=s.index('static inline __attribute__((always_inline)) long run_task_impl(')
end=s.index('// 2. GEMM 段',begin)
part=s[begin:end]
old='    } else {\n        uint64_t t0 = c->prof >= 2 ? cyc() : 0;\n        const svbool_t pg = svptrue_b64();'
assert part.count(old)==1
part=part.replace(old,'    } else if (!(fast_owner && c->preload58)) {\n        uint64_t t0 = c->prof >= 2 ? cyc() : 0;\n        const svbool_t pg = svptrue_b64();')
s=s[:begin]+part+s[end:]
marker='static inline __attribute__((always_inline)) long run_task_impl('
helper='''// Never read another thread's unsolved rows. Each interval is consumed once
// within this invocation, after any earlier solved prefix has been copied.
static __attribute__((noinline)) void preload_owner58(const ctx_t* c,tbuf_t* tb,long p,long begin,long end)
{
    uint64_t t0=c->prof>=2?cyc():0;
    const svbool_t pg=svptrue_b64();
    for(long r=begin;r<end;r++){
        if(c->loadpf>0 && r+c->loadpf<end){
            const double* future=c->B+(r+c->loadpf)*c->ldb+p*96;
            for(long j=0;j<96;j+=8)__builtin_prefetch(future+j,0,3);
        }
        const double* src=c->B+r*c->ldb+p*96;
        double* dst=tb->cbuf+r*96;
        for(long j=0;j<96;j+=8)svst1_f64(pg,dst+j,svld1_f64(pg,src+j));
    }
    if(c->prof>=2)tb->t_load+=cyc()-t0;
}
'''
s=s.replace(marker,helper+'\n'+marker)
old='                    const int fullpan = !c.cshift || (tsk->p > 0 && tsk->p < c.npan - 1);'
assert s.count(old)==1
s=s.replace(old,old+'''
                    if(c.preload58 && (s==0 || sched[s-1]/c.nrb!=tsk->p)){
                        long last=s;
                        while(last+1<k && sched[last+1]/c.nrb==tsk->p)last++;
                        preload_owner58(&c,&g_tb[tid],tsk->p,tsk->i*64,(sched[last]%c.nrb+1)*64);
                    }''')
(out/'trsm-preload58.c').write_text(s)
driver=(out/'independent-owner58.c').read_text().replace('trsm-ownerdiag58.c','trsm-preload58.c')
(out/'independent-preload58.c').write_text(driver)
