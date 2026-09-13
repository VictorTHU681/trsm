#!/usr/bin/env python3
from pathlib import Path
import subprocess
here=Path(__file__).resolve().parent
root=here.parents[1]
out=here/'generated'
s=subprocess.check_output(['git','show','3b24f38:trsm.c'],cwd=root).decode()
start=s.index('static void diag_prep_sub(')
end=s.index('// ---------------------------------------------------------------- 上下文',start)
reference=s[start:end].replace('diag_prep_sub(', 'diag_prep_reference(',1)
old='''    for (int c = 0; c < 16; c++) {
        Linv[c][c] = 1.0 / Lss[c][c];
        for (int r = c + 1; r < 16; r++) {
            double acc = 0.0;
            for (int q = c; q < r; q++) acc += Lss[r][q] * Linv[q][c];
            Linv[r][c] = -acc / Lss[r][r];
        }
    }'''
new='''    if(g_diag_order==1 || g_diag_order==2){
        // Increasing q preserves each coefficient's accumulation order; the
        // independent columns are adjacent and can be evaluated as vectors.
        for(int r=0;r<16;r++){
            double accum[16]={0};
            double inverse=1.0/Lss[r][r];
            for(int q=0;q<r;q++){
                double a=Lss[r][q];
                for(int col=0;col<=q;col++)accum[col]+=a*Linv[q][col];
            }
            for(int col=0;col<r;col++)
                Linv[r][col]=g_diag_order==2?-accum[col]*inverse:-accum[col]/Lss[r][r];
            Linv[r][r]=inverse;
        }
    }else{
        double inverses[16];
        if(g_diag_order==3)for(int r=0;r<16;r++)inverses[r]=1.0/Lss[r][r];
        for (int c = 0; c < 16; c++) {
            Linv[c][c] = g_diag_order==3?inverses[c]:1.0/Lss[c][c];
            for (int r = c + 1; r < 16; r++) {
                double acc = 0.0;
                for (int q = c; q < r; q++) acc += Lss[r][q] * Linv[q][c];
                Linv[r][c] = g_diag_order==3?-acc*inverses[r]:-acc/Lss[r][r];
            }
        }
    }'''
assert s.count(old)==1
s=s.replace(old,new)
s=s.replace('static void diag_prep_sub(', 'static int g_diag_order;\nstatic void diag_prep_sub(',1)
marker='    c.prof = env_int("TRSM_PROFILE", 0);'
assert s.count(marker)==1
s=s.replace(marker,marker+'\n    g_diag_order=env_int("TRSM_DIAG_ORDER",0);')
(out/'trsm-diag-order.c').write_text(s)
(out/'diag-reference.h').write_text(reference)
(out/'diag_check.c').write_bytes((here/'diag_check.c').read_bytes())
