#!/usr/bin/env python3
from pathlib import Path
out=Path(__file__).resolve().parent/'generated'
s=(out/'trsm-fused-final.c').read_text()
key='static long run_task_owner_cursor(const ctx_t* c,tbuf_t* tb,const task_t* tk,const task_t* guess,int tid)'
assert s.count(key)==1
s=s.replace(key,'''#ifndef TRSM_OWNER_ALIGN
#define TRSM_OWNER_ALIGN 64
#endif
static __attribute__((aligned(TRSM_OWNER_ALIGN))) long run_task_owner_cursor(const ctx_t* c,tbuf_t* tb,const task_t* tk,const task_t* guess,int tid)''')
(out/'trsm-fused-align.c').write_text(s)
