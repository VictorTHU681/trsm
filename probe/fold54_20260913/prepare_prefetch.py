#!/usr/bin/env python3
from pathlib import Path
here=Path(__file__).resolve().parent
out=here/'generated'
s=(out/'trsm-fold-kernels.c').read_text()
def replace(old,new):
    global s
    assert s.count(old)==1,(old,s.count(old))
    s=s.replace(old,new)
def replace_run(old,new):
    global s
    start=s.index('static inline __attribute__((always_inline)) long run_task_impl(')
    end=s.index('static long run_task(',start)
    body=s[start:end]
    assert body.count(old)==1,(old,body.count(old))
    s=s[:start]+body.replace(old,new)+s[end:]
replace('static int g_fold_owner;','static int g_fold_owner,g_fold_pref,g_fold_nopack;')
replace('    g_fold_kernel=env_int("TRSM_FOLD_KERNEL",0);',
'''    g_fold_pref=env_int("TRSM_FOLD_PREF",0);
    g_fold_nopack=env_int("TRSM_FOLD_NOPACK",1);
    g_fold_kernel=env_int("TRSM_FOLD_KERNEL",0);''')
marker='static void pack_l_chunked(const ctx_t* c, long g) {'
replace(marker,marker+'\n    if(g_fold_owner && g_fold_nopack)return; // Only the freshly transformed coefficients are consumed.')
replace('            if (fast_owner && c->nexta && tkn && tkn->kind == 0 && tkn->i > 0) {',
        '            if (fast_owner && fast_owner!=3 && c->nexta && tkn && tkn->kind == 0 && tkn->i > 0) {')
marker='''            // 非临时存储按 X 行的 cache 行对齐：头部分行（谓词）+ 整行 + 尾部分行。bench 的 B 只 16 B 对齐，'''
replace_run(marker,'''            const double* fold_next_b=NULL;
            const char* fold_next_a=NULL;
            long fold_a_lines=0;
            if(fast_owner==3 && tkn && tkn->kind==0){
                long next_i0=(long)tkn->i*64;
                if(g_fold_pref&1)fold_next_b=c->B+next_i0*ldb+pan_c0(c,tkn->p);
                if(g_fold_pref&2){
                    fold_next_a=(const char*)(g_fold_coeff+diag_strip_off((int)(next_i0/16)));
                    fold_a_lines=(next_i0+16)*2;
                }
            }
'''+marker)
marker='''                if (na) {
                    const char* q = na + r * nal;'''
replace(marker,'''                if(fold_next_b){
                    const char* nb=(const char*)(fold_next_b+r*ldb);
                    for(int line=0;line<12;line++)__builtin_prefetch(nb+64*line,0,2);
                }
                if(fold_next_a)for(long line=r;line<fold_a_lines;line+=64)
                    __builtin_prefetch(fold_next_a+64*line,0,2);
'''+marker)
(out/'trsm-fold-prefetch.c').write_text(s)
