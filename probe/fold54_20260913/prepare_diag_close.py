#!/usr/bin/env python3
from pathlib import Path
import subprocess
here=Path(__file__).resolve().parent
root=here.parents[1]
out=here/'generated'
s=(out/'trsm-diag-final-audit.c').read_text()
marker='static int g_diag_sme,g_diag_row;'
assert s.count(marker)==1
s=s.replace(marker,marker+'\nstatic long audit_sme_transforms;')
marker='''    double at[256],bt[256];'''
assert s.count(marker)==1
s=s.replace(marker,'    __atomic_fetch_add(&audit_sme_transforms,1,__ATOMIC_RELAXED);\n'+marker)
s=s.replace('    audit_diag_count=audit_pack_count=0;',
            '    audit_diag_count=audit_pack_count=0;\n    audit_sme_transforms=0;')
s=s.replace('    long expected_pack=0;', '''    long expected_sme=g_diag_sme?ndiag_items-c.nrb:0;
    int sme_ok=audit_sme_transforms==expected_sme;
    printf("DIAG_SME_AUDIT m=%ld n=%ld calls=%ld expected=%ld %s\\n",m,n,audit_sme_transforms,expected_sme,sme_ok?"PASS":"FAIL");
    if(!sme_ok)abort();
    long expected_pack=0;''')
(out/'trsm-diag-close-audit.c').write_text(s)
for kind in ('fold','duplex'):
    s=(out/f'independent-{kind}.c').read_text()
    s=s.replace('            l_trsm(m,n,L,lda,B,ldb);',
                '            setenv("TRSM_DIAG_SME",reuse?"0":"1",1);\n            l_trsm(m,n,L,lda,B,ldb);')
    s=s.replace('mode==1?1e100:mode==2?1e-100',
                'mode==1?0x1p1000:mode==2?0x1p-1000')
    (out/f'independent-diag-extreme-{kind}.c').write_text(s)
s=(out/'gen-strict.c').read_text()
s=s[:s.index('int main(int argc, char** argv)')]+r'''
int main(void)
{
    const int shapes[][2]={{4095,17},{4096,17},{4097,17},{4160,128}};
    int failures=0;
    for(int build=0;build<2;build++)for(int mode=0;mode<3;mode++){
        setenv("TRSM_TBUILD",build?"1":"0",1);
        if(mode==0)unsetenv("TRSM_DIAG_SME");else setenv("TRSM_DIAG_SME",mode==1?"0":"1",1);
        for(int shape=0;shape<4;shape++){
            int m=shapes[shape][0],n=shapes[shape][1];
            failures+=run_case(m,n,m+7,n+11,17391+shape);
        }
    }
    printf("DIAG_GATE %s: %d failures\n",failures?"FAIL":"PASS",failures);
    return failures?1:0;
}
'''
(out/'diag-gate.c').write_text(s)
