#!/usr/bin/env python3
from pathlib import Path
here=Path(__file__).resolve().parent
out=here/'generated'
s=(out/'gemm-duplex-profile.c').read_text()
s=s.replace('(ch*strips+strip)*1024','((strip/(mb/16))*chunks*(mb/16)+ch*(mb/16)+strip%(mb/16))*1024')
s=s.replace('(size_t)strips*chunks*1024*8','(size_t)blocks*(mb/16)*chunks*1024*8')
s=s.replace('(ch*strips+ib*(mb/16)+r)*1024','(ib*chunks*(mb/16)+ch*(mb/16)+r)*1024')
s=s.replace('((ch+1)*strips+ib*(mb/16))*1024','(ib*chunks*(mb/16)+(ch+1)*(mb/16))*1024')
s=s.replace('long slot=r*NT+t;','long first=S-(32+NT-1)/NT; if(first<0)first=0;\n                        long slot=(r-first)*NT+t;')
s=s.replace('Bnext&&slot<32','Bnext&&slot>=0&&slot<32')
s=s.replace('kloop_duplex_bank','kloop_hoist_bank')
(out/'gemm-duplex-layout.c').write_text(s)
