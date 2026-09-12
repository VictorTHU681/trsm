#!/usr/bin/env python3
from pathlib import Path
HERE=Path(__file__).resolve().parent
OUT=HERE/'generated'
s=(HERE/'strassen_pilot.c').read_text()
(OUT/'strassen_pilot.c').write_text(s)
s=s.replace('#include "kblas.h"','#include "kblas.h"\nvoid pilot_gemm(int,int,int,const double*,int,const double*,int,double*,int);',1)
old='        cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,m,n,k,1,a,lda,b,ldb,0,c,ldc);'
assert old in s
s=s.replace(old,'        pilot_gemm(m,n,k,a,lda,b,ldb,c,ldc);',1)
(OUT/'strassen-duplex.c').write_text(s)
(OUT/'gemm_duplex.c').write_bytes((HERE/'gemm_duplex.c').read_bytes())
