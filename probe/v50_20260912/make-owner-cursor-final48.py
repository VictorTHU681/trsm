from pathlib import Path
p=Path(__file__).resolve().parent;s=(p/'trsm-owner-hoist48.c').read_text();old=(p/'trsm-hoist-pf45.c').read_text()
for bank in (0,4):
 a=s.index(f'static inline __attribute__((always_inline)) void kloop_owner_hoist{bank}(const double* a');b=s.index('PCLOB);}',a)+len('PCLOB);}')
 x=old.index(f'static inline __attribute__((always_inline)) void kloop_owner_bank{bank}(const double* a');y=old.index('PCLOB);}',x)+len('PCLOB);}')
 new=old[x:y].replace(f'kloop_owner_bank{bank}',f'kloop_owner_cursor{bank}')
 s=s[:a]+new+s[b:]
 s=s.replace(f'kloop_owner_hoist{bank}',f'kloop_owner_cursor{bank}')
s=s.replace('g_owner_hoist','g_owner_cursor').replace('TRSM_OWNER_HOIST','TRSM_OWNER_CURSOR').replace('run_task_owner_hoist','run_task_owner_cursor')
s=s.replace('env_int("TRSM_OWNER_CURSOR",0)','env_int("TRSM_OWNER_CURSOR",1)')
(p/'trsm-owner-cursor-final48.c').write_text(s)
check='''    int expected_owner=env_int("TRSM_EXPECT_OWNER",-1);
    if(expected_owner>=0 && (c.special==5)!=expected_owner)abort();
    int expected_special=env_int("TRSM_EXPECT_SPECIAL",-1);
    if(expected_special>=0 && c.special!=expected_special)abort();
    int expected_cursor=env_int("TRSM_EXPECT_CURSOR",-1);
    if(expected_cursor>=0 && (g_owner_cursor && c.special==5)!=expected_cursor)abort();
'''
s=s.replace('    g_align_owner_buffers =',check+'    g_align_owner_buffers =',1);(p/'trsm-owner-cursor-final48-audit.c').write_text(s)
s=(p/'independent-owner-switch.c').read_text().replace('setenv("TRSM_OWNER",reuse?"0":"1",1);','setenv("TRSM_OWNER","1",1);\n            setenv("TRSM_OWNER_CURSOR",reuse?"0":"1",1);\n            setenv("TRSM_EXPECT_CURSOR",reuse?"0":"1",1);').replace('setenv("TRSM_EXPECT_OWNER",reuse?"0":"1",1);','setenv("TRSM_EXPECT_OWNER","1",1);');(p/'independent-cursor-switch.c').write_text(s)
s=(p/'v47-boundary.c').read_text().replace('trsm-v47-codex.c','trsm-owner-cursor-final48.c').replace('kloop_owner_bank0','kloop_owner_cursor0').replace('kloop_owner_bank4','kloop_owner_cursor4');(p/'owner-cursor-final48-boundary.c').write_text(s)
