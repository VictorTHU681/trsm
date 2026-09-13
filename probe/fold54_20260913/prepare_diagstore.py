#!/usr/bin/env python3
from pathlib import Path
import subprocess
here=Path(__file__).resolve().parent
root=here.parents[1]
out=here/'generated'
s=subprocess.check_output(['git','show','36a092c:trsm.c'],cwd=root).decode()
start=s.index('#define OWNER_DSTORE_ROW(')
end=s.index('#undef OWNER_DSTORE_ROWS',start)+len('#undef OWNER_DSTORE_ROWS')
helper=s[start:end].replace('OWNER_DSTORE','SPECIAL_DSTORE').replace('za_owner_dstore','za_special_dstore').replace('#768','#1024')
marker='static inline __attribute__((always_inline)) long run_task_special_impl('
s=s.replace(marker,'static int g_diag_store,g_diag_store_max;\n'+helper+'\n'+marker)
start=s.index(marker);end=s.index('static long run_task_special1(',start)
body=s[start:end]
needle='''                if (pend && K >= 65) {
                    PROF_T(c, kloop_epi_set(A, 128, 128, cb + 32 * t, cs, K, cols, &pf, scratch, pend_c, cs, pend_cols), t_dkloop);'''
assert body.count(needle)==1
body=body.replace(needle,'''                if(g_diag_store && (!g_diag_store_max || K<=g_diag_store_max)){
                    if(pend){PROF_T(c,epi_set_std(scratch,pend_c,cs,pend_cols),t_depi);pend=0;}
                    PROF_T(c,kloop_plain(A,128,128,cb+32*t,cs,K,cols,&pf),t_dkloop);
                    if(g_diag_store==1){PROF_T(c,za_store_fixed(cb+16L*s*ncp+32*t,cs,cols),t_ddump);}
                    else{PROF_T(c,za_special_dstore(cb+16L*s*ncp+32*t),t_ddump);}
                    if(c->prof>=2){tb->n_kstep+=K;tb->n_dkstep+=K;}
                    continue;
                }
'''+needle)
s=s[:start]+body+s[end:]
marker='    g_owner_cursor=env_int('
s=s.replace(marker,'''    g_diag_store=env_int("TRSM_DIAG_STORE",0);
    g_diag_store_max=env_int("TRSM_DIAG_STORE_MAX",64);
'''+marker)
(out/'trsm-diagstore.c').write_text(s)
(out/'independent-duplex.c').write_bytes(subprocess.check_output(['git','show','36a092c:probe/v50_20260912/independent-duplex43.c'],cwd=root))
