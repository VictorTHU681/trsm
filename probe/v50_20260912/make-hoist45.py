from pathlib import Path
import re,ast,json
p=Path(__file__).resolve().parent;s=(p/'trsm-v45-codex.c').read_text()
def transform(source, family, ab, pf):
 for bank in (0,4):
  name=f'kloop_{family}_bank{bank}'
  # First occurrence with named arguments is definition rather than prototype.
  start=source.index(f'static inline __attribute__((always_inline)) void {name}(const double* a')
  end=source.index('PCLOB);}',start)+len('PCLOB);}')
  fun=source[start:end];beg=fun.index('asm volatile(')+len('asm volatile(');fin=fun.index(': [a]')
  part=fun[beg:fin]
  bs=768 if family=='owner' else 128;cs=bs if family=='owner' else 1024
  for suffix in ['BODY','NEXT','LAST','TAIL']:
   key=f'DUPLEX{bank}_U_{suffix}'
   match=re.search(r'#define '+key+r'(\([^\n]*?\))? \\\n((?:[^\n]*\\\n)*[^\n]*)',s);assert match
   body=match.group(2).replace(' \\\n','\n').rstrip()
   for k,v in {'BS':str(bs),'CS':str(cs),'C8':str(cs*8),'C864':str(cs*8+64)}.items():body=body.replace('" #'+k+' "',v)
   body=body.replace('ALD ', '"ldnt1d " ' if family=='owner' or bank==4 else '"ld1d " ')
   part=re.sub(key+(r'\([^)]*\)' if match.group(1) else ''),lambda _:body,part)
  asm=''.join(ast.literal_eval(x) for x in re.findall(r'"(?:\\.|[^"\\])*"',part))
  lines=[x.strip() for x in asm.splitlines()];out=[];old={x:0 for x in ('a','b','ap','bp','pc')};cur={'a':0,'b':0};loads={'a':[],'b':[]};removed=0
  for line in lines:
   m=re.fullmatch(r'add %\[(a|b|ap|bp|pc)\], %\[\1\], #(\d+)',line)
   if m and ((ab and m[1] in ('a','b')) or (pf and m[1] in ('ap','bp','pc'))):old[m[1]]+=int(m[2]);removed+=1;continue
   m=re.fullmatch(r'(ld(?:nt)?1d z\d+.d, p0/z, )\[%\[(a|b)\](?:, #(-?\d+), mul vl)?\]',line)
   if m and ab:
    ptr=m[2];addr=old[ptr]+int(m[3] or 0)*64
    stride=bs if ptr=='b' else 128;group=2 if stride==768 else 8
    base=(old[ptr]//(stride*group))*stride*group+(384 if stride==768 else 512)
    if base!=cur[ptr]:out.append(f'add %[{ptr}], %[{ptr}], #{base-cur[ptr]}');cur[ptr]=base
    off=(addr-base)//64;assert -8<=off<=7,(name,line,addr,base,off)
    line=m[1]+f'[%[{ptr}], #{off}, mul vl]';loads[ptr].append(addr)
   m=re.fullmatch(r'(prfm pldl2keep, )\[%\[(ap|bp|pc)\](?:, #(\d+))?\]',line)
   if m and pf:
    off=old[m[2]]+int(m[3] or 0);assert off%8==0 and 0<=off<32768
    line=m[1]+f'[%[{m[2]}], #{off}]'
   out.append(line)
  if ab:
   for ptr in ('a','b'):
    stride=bs if ptr=='b' else 128
    assert loads[ptr]==[i*stride+j*64 for i in range(64) for j in range(2)],(name,ptr)
  print(name,'ab',ab,'pf',pf,'instructions',len(lines),'->',len(out),'old_add_removed',removed)
  new=fun[:beg]+'\n'+''.join(' '+json.dumps(l+'\n\t')+'\n' for l in out)+fun[fin:]
  source=source[:start]+new+source[end:]
 return source
for tag,ab,pf in [('ab',1,0),('pf',0,1),('all',1,1)]:
 out=transform(transform(s,'duplex',ab,pf),'owner',ab,pf)
 (p/f'trsm-hoist-{tag}45.c').write_text(out)
