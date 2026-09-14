#!/usr/bin/env python3
from pathlib import Path
here=Path(__file__).resolve().parent;out=here/'generated'
base=(out/'trsm-v55.c').read_text()
begin=base.index('#define KLOOP_REST');end=base.index('// 托管回写走线器',begin)
rest=base[begin:end].replace('KS_ODD PFC','KS_ODD')
assert rest!=base[begin:end]
s=(out/'trsm-diagfixed.c').read_text()
marker='DEF_DIAG57_PLAIN_2(diag57_plain_2)'
assert s.count(marker)==1
s=s.replace(marker,'#pragma push_macro("KLOOP_REST")\n#undef KLOOP_REST\n'+rest+'\n'+marker)
marker='DEF_DIAG57_EPI64_2(diag57_epi64_2, EV_SET)'
s=s.replace(marker,marker+'\n#pragma pop_macro("KLOOP_REST")')
(out/'trsm-diagfixed-prefetch.c').write_text(s)
s=(out/'trsm-ownerdiag.c').read_text()
marker='DEF_OWNERDIAG57(ownerdiag57_plain)'
assert s.count(marker)==1
s=s.replace(marker,marker+'\n#pragma push_macro("PFC")\n#undef PFC\n#define PFC\nDEF_OWNERDIAG57(ownerdiag57_nocpf)\n#pragma pop_macro("PFC")')
call='g_owner_diag_simple ? ownerdiag57_plain(A,128,128,cb+32*t,cs,K,cols,&pf)'
assert s.count(call)==1
s=s.replace(call,'g_owner_diag_simple==2 ? ownerdiag57_nocpf(A,128,128,cb+32*t,cs,K,cols,&pf) : '+call)
old='g_owner_diag_simple=c.special==5 && c.ddirect && !c.swhost && env_int("TRSM_OWNER_DIAG_SIMPLE",0);'
assert s.count(old)==1
s=s.replace(old,'g_owner_diag_simple=c.special==5 && c.ddirect && !c.swhost ? env_int("TRSM_OWNER_DIAG_SIMPLE",0) : 0;')
(out/'trsm-ownerdiag-prefetch.c').write_text(s)
