from pathlib import Path
import re,ast,json
p=Path(__file__).resolve().parent;src=(p/'trsm-v47-codex.c').read_text();s=src
protos=''
for bank in (0,4):
 name=f'kloop_owner_bank{bank}';newname=f'kloop_owner_import{bank}'
 start=s.index(f'static inline __attribute__((always_inline)) void {name}(const double* a');end=s.index('PCLOB);}',start)+len('PCLOB);}')
 fun=s[start:end];beg=fun.index('asm volatile(')+len('asm volatile(');fin=fun.index(': [a]');part=fun[beg:fin]
 for suffix in ['BODY','NEXT','LAST','TAIL']:
  key=f'DUPLEX{bank}_U_{suffix}';m=re.search(r'#define '+key+r'(\([^\n]*?\))? \\\n((?:[^\n]*\\\n)*[^\n]*)',src);assert m
  body=m.group(2).replace(' \\\n','\n').rstrip()
  for k,v in {'BS':'768','CS':'768','C8':'6144','C864':'6208'}.items():body=body.replace('" #'+k+' "',v)
  body=body.replace('ALD ','"ldnt1d " ')
  part=re.sub(key+(r'\([^)]*\)' if m.group(1) else ''),lambda _:body,part)
 asm=''.join(ast.literal_eval(x) for x in re.findall(r'"(?:\\.|[^"\\])*"',part));lines=[x.strip() for x in asm.splitlines()]
 count=0;out=[];copies=0
 for l in lines:
  out.append(l)
  if l.startswith('fmopa '):
   count+=1
   if count>=20 and count<=240 and count%20==0:
    j=count//20-1;out.append(f'ld1d z24.d,p0/z,[%[isrc],#{j-8},mul vl]')
   if count>=24 and count<=244 and count%20==4:
    j=(count-4)//20-1;out += [f'st1d z24.d,p0,[%[idst],#{j-8},mul vl]',f'prfm pldl2keep,[%[ipf],#{j*64}]'];copies+=1
 assert count==256 and copies==12
 head=fun[:beg].replace(name,newname).replace('const char* pc)','const char* pc, const double* isrc, double* idst, const char* ipf)').replace('    asm volatile(', '    isrc += 64; idst += 64;\n    asm volatile(')
 tail=fun[fin:].replace(':: "memory"',': [isrc] "r"(isrc), [idst] "r"(idst), [ipf] "r"(ipf): "memory"')
 assert '[isrc]' in tail
 new=head+'\n'+''.join(' '+json.dumps(l+'\n\t')+'\n' for l in out)+tail
 s=s[:end]+'\n'+new+s[end:]
 protos+=f'static inline __attribute__((always_inline)) void {newname}(const double*, const double*, double*, const char*, const char*, const char*, const double*, double*, const char*);\n'
s=s.replace('// Full 16x32 diagonal results',protos+'static int g_oimport;\n\n// Full 16x32 diagonal results',1)
s=s.replace('    double* cbuf;    // mb*nc','    long import_p, import_i, import_rows;\n    double* cbuf;    // mb*nc',1)
s=s.replace('    // 1. C 块 ← B 子块', '''    long imported = fast_owner && tb->import_p==p && tb->import_i==i ? tb->import_rows : 0;
    if(fast_owner)tb->import_rows=0;
    const int can_import=fast_owner && g_oimport && tkn && tkn->p==p && tkn->i==i+1;
    long import_rows=0;
    const double* import_src=can_import?c->B+(i0+64)*ldb+c0:NULL;
    double* import_dst=can_import?tb->cbuf+(i0+64)*96:NULL;
    if(can_import)for(int v=0;v<12;v++)__builtin_prefetch(import_src+8*v,0,2);

    // 1. C 块 ← B 子块''',1)
s=s.replace('for (long r = 0; r < 16L * S; r++) {','for (long r = imported; r < 16L * S; r++) {',1)
a=s.index('                PROF_T(c, kloop_owner_bank0(');b=s.index('                pend = 1;',a)
old=s[a:b]
new=''
for bank in (0,4):
 call=('A, Xk + 16*t, pend ? pend_c : cur, pf.p, pf.p2, (const char*)cur' if bank==0 else 'A+1024, Xk+16*t, cur, pf.p+1024, pf.p2+1024, (const char*)(cur+16*ncp)')
 new+=f'''                if(can_import && import_rows<64){{
                    const double* sp=import_src+import_rows*ldb;
                    double* dp=import_dst+import_rows*96;
                    const char* ipf=(const char*)(sp+(import_rows<63?ldb:0));
                    PROF_T(c,kloop_owner_import{bank}({call},sp,dp,ipf),t_kloop);
                    import_rows++;
                }} else {{PROF_T(c,kloop_owner_bank{bank}({call}),t_kloop);}}
'''
s=s[:a]+new+s[b:]
s=s.replace('    __atomic_store_n(&c->flags[i * c->npan + p], 1, __ATOMIC_RELEASE);', '''    if(fast_owner && can_import){tb->import_p=p;tb->import_i=i+1;tb->import_rows=import_rows;}
    __atomic_store_n(&c->flags[i * c->npan + p], 1, __ATOMIC_RELEASE);''',1)
s=s.replace('    c.prof = env_int("TRSM_PROFILE", 0);','    c.prof = env_int("TRSM_PROFILE", 0);\n    g_oimport=env_int("TRSM_OIMPORT",1);',1)
s=s.replace('        g_tb[tid].swn = g_tb[tid].swi = 0;','        g_tb[tid].import_rows=0;\n        g_tb[tid].swn = g_tb[tid].swi = 0;',1)
(p/'trsm-owner-import47.c').write_text(s)
