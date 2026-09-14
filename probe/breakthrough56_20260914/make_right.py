#!/usr/bin/env python3
from pathlib import Path
here=Path(__file__).resolve().parent;out=here/'generated'
s=(out/'fullinv-compact.h').read_text()
s=s.replace('inv_compact','right_compact').replace('fmopa','fmops')
(out/'right-kernel.h').write_text(s)
(out/'rightlooking.c').write_text((here/'rightlooking.c').read_text())
