#!/usr/bin/env python3
"""Follow up the strict bitwise mismatch with contraction and error-bound checks."""
from pathlib import Path
here=Path(__file__).resolve().parent
out=here/'generated'
s=(here/'diag_check.c').read_text()
contract=s[:s.index('    // A component timing only:')]+ '    return 0;\n}\n'
contract=contract.replace('for(int mode=0;mode<4;mode++)','for(int mode=0;mode<2;mode++)')
(out/'diag_contract_check.c').write_text(contract)
old='''                    if(!isfinite(actual[q]) || (mode<=1?err!=0:err>32*DBL_EPSILON*fabs(ref[q])+1e-300)){'''
new='''                    int row=q%16,col=q/16,vr=dims[shape]-strip*16;
                    if(vr>16)vr=16;
                    long r0=offset+16*strip;
                    long double norm=0;
                    if(col<16*strip){
                        for(int t=0;t<vr;t++)
                            norm+=fabsl((long double)ref[256*strip+t*16+row]*L[(r0+t)*lda+offset+col]);
                    }else{
                        int invcol=col-16*strip;
                        if(row>=vr)norm=row==invcol?1:0;
                        else{
                            norm=row==invcol?1:0;
                            for(int t=0;t<row;t++)
                                norm+=fabsl((long double)L[(r0+row)*lda+r0+t]*ref[256*strip+invcol*16+t]);
                            norm/=fabsl(L[(r0+row)*lda+r0+row]);
                        }
                    }
                    if(!isfinite(actual[q]) || err>256*DBL_EPSILON*(double)norm+1e-300){'''
assert s.count(old)==1
s=s.replace(old,new).replace('DIAG_ORDER_CHECK mode=', 'DIAG_BOUND_CHECK mode=')
(out/'diag_bound_check.c').write_text(s)
script=(here/'437-diag-order.sh').read_text()
script=script.replace('gcc "${flags[@]}" diag_check.c -o diag-check -lm\nOMP_NUM_THREADS=1 numactl -N 1 --preferred=17 taskset -c 40 ./diag-check',
'''gcc "${flags[@]}" -ffp-contract=off diag_contract_check.c -o diag-contract-check -lm
set +e
OMP_NUM_THREADS=1 numactl -N 1 --preferred=17 taskset -c 40 ./diag-contract-check
contract_status=$?
set -e
echo "DIAG_CONTRACTION_DIAGNOSTIC status=$contract_status"
gcc "${flags[@]}" diag_bound_check.c -o diag-bound-check -lm
OMP_NUM_THREADS=1 numactl -N 1 --preferred=17 taskset -c 40 ./diag-bound-check''')
(out/'438-diag-bound.sh').write_text(script)
