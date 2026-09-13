#!/usr/bin/env python3
from pathlib import Path
here=Path(__file__).resolve().parent
out=here/'generated'
s=(out/'trsm-fold-prefetch.c').read_text()
old='''else if(fast_owner==3 && g_fold_kernel==3){PROF_T(c,fold_keep(A,source,K),t_dkloop);}'''
new='''else if(fast_owner==3 && g_fold_kernel==3){
                        if(g_fold_lastnt && t+1==ntile){PROF_T(c,fold_nt(A,source,K),t_dkloop);}
                        else{PROF_T(c,fold_keep(A,source,K),t_dkloop);}
                    }'''
assert s.count(old)==1
s=s.replace(old,new).replace('static int g_fold_kernel;', 'static int g_fold_kernel,g_fold_lastnt;')
s=s.replace('    g_fold_kernel=env_int("TRSM_FOLD_KERNEL",0);',
            '    g_fold_lastnt=env_int("TRSM_FOLD_LASTNT",0);\n    g_fold_kernel=env_int("TRSM_FOLD_KERNEL",0);')
(out/'trsm-fold-last.c').write_text(s)
