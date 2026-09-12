#!/usr/bin/env python3
from pathlib import Path
import json
here=Path(__file__).resolve().parent
out=here/'generated'
s=(out/'gemm-duplex-tasks.c').read_text()
helper=''
for name,store in [('dense_clear','st1d'),('dense_clear_nt','stnt1d')]:
    lines=['ptrue p0.d','dup z0.d, #0','1:',f'{store} z0.d, p0, [%[c]]','add %[c], %[c], #64','subs %[bytes], %[bytes], #64','b.ne 1b']
    helper+=f'''static inline __attribute__((always_inline)) void {name}(double* c,long bytes) {{
    asm volatile(
'''+''.join(' '+json.dumps(x+'\n\t')+'\n' for x in lines)+'''
    : [c]"+&r"(c),[bytes]"+&r"(bytes) : : "memory","cc","v0","p0");
}
'''
s=s.replace('void pilot_gemm(',helper+'\nvoid pilot_gemm(',1)
s=s.replace('const int task_order=', 'const int clear_mode=env_int("DENSE_CLEAR",0);\n    const int task_order=',1)
s=s.replace('memset(cb,0,(size_t)mb*nc*8);','if(!clear_mode)memset(cb,0,(size_t)mb*nc*8);',1)
s=s.replace('sme_start();sme_zero_za();','''sme_start();sme_zero_za();
            if(clear_mode){
                tz=cyc();
                if(clear_mode==1)dense_clear(cb,mb*nc*8);
                else dense_clear_nt(cb,mb*nc*8);
                zr+=cyc()-tz;
            }''',1)
(out/'gemm-duplex-clear.c').write_text(s)
