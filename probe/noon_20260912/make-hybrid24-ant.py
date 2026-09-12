from pathlib import Path
import re
p=Path(__file__).resolve().parent
s=(p/'hybrid24-duplex.c').read_text()
s=s.replace('#include "trsm-v47-codex.c"','#include "trsm-v47-codex.c"\n#ifndef H24_ANT\n#define H24_ANT 0\n#endif\n#if H24_ANT==2\n#define H24_A0 "ldnt1d "\n#else\n#define H24_A0 "ld1d "\n#endif\n#if H24_ANT>=1\n#define H24_A4 "ldnt1d "\n#else\n#define H24_A4 "ld1d "\n#endif',1)
a=s.index('static inline __attribute__((always_inline)) void hduplex4(')
x,y=s[:a],s[a:]
x=re.sub(r'"ld1d (z\d+\.d,p0/z,\[%\[a\][^\n]+)',r'H24_A0 "\1',x)
y=re.sub(r'"ld1d (z\d+\.d,p0/z,\[%\[a\][^\n]+)',r'H24_A4 "\1',y)
s=x+y
s=s.replace('int main(void){','int main(void){printf("H24_ANT=%d\\n",H24_ANT);',1)
(p/'hybrid24-duplex-ant.c').write_text(s)
