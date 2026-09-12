from pathlib import Path
p=Path(__file__).resolve().parent
s=(p/'pmu_insert.txt').read_text();pmu=(p.parent/'noon_20260912'/'pmu_support.h').read_text()
def patch(t):
 t=t.replace('// ---------------------------------------------------------------- 主流程',pmu+'// ---------------------------------------------------------------- 主流程',1)
 t=t.replace('    c.prof = env_int("TRSM_PROFILE", 0);','    c.prof = env_int("TRSM_PROFILE", 0);\n    pmu_on=env_int("TRSM_PMUSOLVE",1);if(pmu_on)pmu_init();',1)
 t=t.replace('        uint64_t tp0 = c.prof >= 2 ? cyc() : 0;','        pmu_begin(tid);\n        uint64_t tp0 = c.prof >= 2 ? cyc() : 0;',1)
 t=t.replace('        if (c.prof >= 2) g_tail_cyc[tid] = cyc();','        if (c.prof >= 2) g_tail_cyc[tid] = cyc();\n        pmu_end(tid);',1)
 a=s.index('    if(pmu_on) {\n        fprintf(stderr,"PMUSOLVE')
 b=s.index('    free(ditem_i);',a)
 t=t.replace('    free(ditem_i);',s[a:b]+'    free(ditem_i);',1)
 return t
if __name__=='__main__':
 import sys
 for tag in sys.argv[1:]:
  (p/f'trsm-{tag}-pmu.c').write_text(patch((p/f'trsm-{tag}.c').read_text()))
