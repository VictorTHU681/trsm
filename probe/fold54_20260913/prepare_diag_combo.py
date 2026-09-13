#!/usr/bin/env python3
from pathlib import Path
here=Path(__file__).resolve().parent
out=here/'generated'
s=(out/'trsm-diag-sme.c').read_text()
old='''    for (int c = 0; c < 16; c++) {
        Linv[c][c] = 1.0 / Lss[c][c];
        for (int r = c + 1; r < 16; r++) {
            double acc = 0.0;
            for (int q = c; q < r; q++) acc += Lss[r][q] * Linv[q][c];
            Linv[r][c] = -acc / Lss[r][r];
        }
    }'''
new='''    if(g_diag_row){
        for(int r=0;r<16;r++){
            double accum[16]={0};
            double inverse=1.0/Lss[r][r];
            for(int q=0;q<r;q++){
                double a=Lss[r][q];
                for(int col=0;col<=q;col++)accum[col]+=a*Linv[q][col];
            }
            for(int col=0;col<r;col++)
                Linv[r][col]=g_diag_row==2?-accum[col]*inverse:-accum[col]/Lss[r][r];
            Linv[r][r]=inverse;
        }
    }else{
'''+old+'''
    }'''
assert s.count(old)==1
s=s.replace(old,new).replace('static int g_diag_sme;', 'static int g_diag_sme,g_diag_row;')
s=s.replace('    g_diag_sme=env_int("TRSM_DIAG_SME",0);',
            '    g_diag_sme=env_int("TRSM_DIAG_SME",0);\n    g_diag_row=env_int("TRSM_DIAG_ROW",0);')
(out/'trsm-diag-combo.c').write_text(s)
test=(out/'diag-sme-check.c').read_text().replace('"trsm-diag-sme.c"','"trsm-diag-combo.c"')
test=test.replace('mode<2','mode<3').replace('1-pass','2-pass').replace('pass<2','pass<3')
test=test.replace('g_diag_sme=mode','g_diag_sme=1;g_diag_row=mode')
test=test.replace('DIAG_SME','DIAG_COMBO')
(out/'diag-combo-check.c').write_text(test)
