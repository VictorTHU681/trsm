#!/usr/bin/env python3
from pathlib import Path
here=Path(__file__).resolve().parent;out=here/'generated'
s=(out/'right-kernel.h').read_text().split('static inline __attribute__((always_inline)) void inv_copy32')[0]
(out/'right-update.h').write_text(s)
(out/'tiled_right.c').write_text((here/'tiled_right.c').read_text())
