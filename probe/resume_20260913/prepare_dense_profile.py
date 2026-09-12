#!/usr/bin/env python3
from pathlib import Path
HERE=Path(__file__).resolve().parent
OUT=HERE/'generated'
s=(HERE/'gemm_duplex.c').read_text()
s=s.replace('    const long mb=192', '    double begin_time=omp_get_wtime(), a_end=0, b_end=0;\n    uint64_t work[1024]={0}, stores[1024]={0}, zero[1024]={0};\n    int cpus[1024]={0};\n    const long mb=192',1)
s=s.replace('        int tid=omp_get_thread_num();pin_self(tid);','        int tid=omp_get_thread_num();pin_self(tid);\n        cpus[tid]=sched_getcpu();\n        uint64_t wk=0,st=0,zr=0;',1)
s=s.replace('#pragma omp for schedule(static)\n        for(long idx=', '#pragma omp single\n        a_end=omp_get_wtime();\n#pragma omp for schedule(static)\n        for(long idx=',1)
s=s.replace('#pragma omp for schedule(dynamic,1)', '#pragma omp single\n        b_end=omp_get_wtime();\n#pragma omp for schedule(dynamic,1)',1)
s=s.replace('            memset(cb,0,(size_t)mb*nc*8);','            uint64_t tz=cyc();\n            memset(cb,0,(size_t)mb*nc*8);\n            zr+=cyc()-tz;',1)
s=s.replace('            sme_start();sme_zero_za();','            sme_start();sme_zero_za();\n            uint64_t tw=cyc();',1)
s=s.replace('            sme_stop();\n            for(long i=0;i<rows;', '            wk+=cyc()-tw;\n            sme_stop();\n            uint64_t ts=cyc();\n            for(long i=0;i<rows;',1)
s=s.replace('=-cb[i*nc+j];\n        }','=-cb[i*nc+j];\n            st+=cyc()-ts;\n        }\n        work[tid]=wk;stores[tid]=st;zero[tid]=zr;',1)
end=s.rfind('\n}')
s=s[:end]+'''
    double finish=omp_get_wtime(), sw=0,ss=0,sz=0,mx=0;
    for(int t=0;t<nth;t++){sw+=work[t];ss+=stores[t];sz+=zero[t];if(work[t]>mx)mx=work[t];}
    double f=1e3/(double)cyc_hz()/nth;
    fprintf(stderr,"DENSE_PROFILE m=%d n=%d k=%d nth=%d ncpu=%d cpu_first=%d cpu_last=%d apack_setup_ms=%.6f bpack_ms=%.6f compute_region_ms=%.6f kernel_mean_ms=%.6f kernel_max_ms=%.6f output_mean_ms=%.6f zero_mean_ms=%.6f total_ms=%.6f\\n",
            m,n,k,nth,g_ncpu,cpus[0],cpus[nth-1],1e3*(a_end-begin_time),1e3*(b_end-a_end),1e3*(finish-b_end),sw*f,mx*f*nth,ss*f,sz*f,1e3*(finish-begin_time));
'''+s[end:]
(OUT/'gemm-duplex-profile.c').write_text(s)
pmu=(HERE.parents[1]/'probe/noon_20260912/pmu_support.h').read_text()
s=s.replace('void pilot_gemm(',pmu+'\nvoid pilot_gemm(',1)
s=s.replace('    double begin_time=', '    pmu_on=1;pmu_init();\n    double begin_time=',1)
s=s.replace('    const int nth=omp_get_max_threads();','    const int nth=omp_get_max_threads();\n    const size_t cstride=env_int("DENSE_CSTRIDE",0)?((size_t)2<<20)/8:mb*nc;',1)
s=s.replace('(size_t)nth*mb*nc*8','(size_t)nth*cstride*8',1)
s=s.replace('dense_c+(size_t)tid*mb*nc','dense_c+(size_t)tid*cstride',1)
s=s.replace('#pragma omp for schedule(dynamic,1)','        pmu_begin(tid);\n#pragma omp for schedule(dynamic,1)',1)
s=s.replace('        work[tid]=wk;', '        pmu_end(tid);\n        work[tid]=wk;',1)
end=s.rfind('\n}')
s=s[:end]+'''
    fprintf(stderr,"DENSE_PMU m=%d n=%d k=%d stride=%zu",m,n,k,cstride*8);
    for(int j=0;j<6;j++){uint64_t sum=0;for(int t=0;t<nth;t++)sum+=pmu_vals[t][3+j];fprintf(stderr," %s=%lu",pmu_names[j],sum);}
    uint64_t en=0,run=0;for(int t=0;t<nth;t++){en+=pmu_vals[t][1];run+=pmu_vals[t][2];}
    fprintf(stderr," enabled_ns=%lu running_ns=%lu\\n",en,run);
'''+s[end:]
(OUT/'gemm-duplex-pmu.c').write_text(s)
s=(OUT/'strassen-duplex.c').read_text()
s=s.replace('    for(int rep=-1;rep<reps;rep++)for(int order=0;order<3;order++){','    int maxdepth=getenv("STRASSEN_MAXDEPTH")?atoi(getenv("STRASSEN_MAXDEPTH")):2;\n    for(int rep=-1;rep<reps;rep++)for(int order=0;order<=maxdepth;order++){',1)
s=s.replace('int depth=(rep&1)?2-order:order;','int depth=(rep&1)?maxdepth-order:order;',1)
(OUT/'strassen-profile.c').write_text(s)
