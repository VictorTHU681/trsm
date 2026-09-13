#!/usr/bin/env python3
from pathlib import Path
out=Path(__file__).resolve().parent/'generated'
s=(out/'independent-fused.c').read_text()
s=s.replace('reuse<2','reuse<4').replace('(reuse?-1:1)','((reuse&1)?-1:1)')
s=s.replace('setenv("TRSM_FUSED_DIAG",reuse?"0":"2",1);',
            'setenv("TRSM_FUSED_DIAG",reuse==2?"0":"2",1);')
(out/'independent-fused-close.c').write_text(s)
