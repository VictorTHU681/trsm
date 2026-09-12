#!/usr/bin/env python3
"""Keep the duplex specialization active while testing its row block size."""
from pathlib import Path
here=Path(__file__).resolve().parent
out=here/'generated'
base=(out/'trsm-control.c').read_text()
for mb in (64,128,192,256):
    s=base.replace('long mb=mode==2?256:192;', f'long mb=(mode==2||mode==4)?{mb}:192;',1)
    s=s.replace('const long mb = (mode==2?256:192),', f'const long mb = (mode==2||mode==4)?{mb}:192,',1)
    s=s.replace('if(mb==256 && !c.direct && !c.ppmode', f'if(mb=={mb} && !c.direct && !c.ppmode',1)
    s=s.replace('if(mb==192 && !c.direct && c.ppmode==1', f'if(mb=={mb} && !c.direct && c.ppmode==1',1)
    s=s.replace('    g_align_owner_buffers =', '''    if (env_int("TRSM_EXPECT_SPECIAL",-1)>=0 && c.special!=env_int("TRSM_EXPECT_SPECIAL",-1))abort();
    g_align_owner_buffers =''',1)
    (out/f'trsm-special-mb{mb}.c').write_text(s)
