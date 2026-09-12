from pathlib import Path
p=Path(__file__).resolve().parent
helper=(p/'pmu_support.h').read_text()
s=(p/'rns-shortk.c').read_text();idx=s.index('static void run(');s=s[:idx]+helper+s[idx:]
s=s.replace('  #pragma omp for schedule(dynamic,1)','  int pmutid=omp_get_thread_num();pmu_begin(pmutid);\n  #pragma omp for schedule(dynamic,1)',1)
idx=s.index('\nint main(');s=s[:idx].rstrip();assert s.endswith('  }\n }\n}')
s=s[:-len('  }\n }\n}')]+'  }\n  pmu_end(pmutid);\n }\n}\n'
s+=r'''
int main(void) {
 int M=512,N=512,K=2048;size_t pn=(size_t)M*N;
 int8_t*A=aligned_alloc(64,(size_t)RNS_R*M*K),*B=aligned_alloc(64,(size_t)RNS_R*N*K);
 int32_t*C=aligned_alloc(64,(size_t)RNS_R*pn*4);double*out=aligned_alloc(64,pn*8);
 if(!A||!B||!C||!out)return 2;
 pmu_init();int ks[]={64,256,2048};
 for(int t=0;t<3;t++) {
  int KC=ks[t];fill(A,M,K,KC,UINT64_C(0x8f881));fill(B,N,K,KC,UINT64_C(0xfe336));
  pmu_on=0;run(A,B,C,out,M,N,K,128,128,KC,0,1);pmu_on=1;
  for(int rep=0;rep<6;rep++) {
   run(A,B,C,out,M,N,K,128,128,KC,0,1);
   uint64_t sums[9]={0};int nt=omp_get_max_threads(),scaled=0;
   for(int tid=0;tid<nt;tid++){if(pmu_vals[tid][0]!=6)return 9;if(pmu_vals[tid][1]!=pmu_vals[tid][2])scaled++;for(int j=1;j<9;j++)sums[j]+=pmu_vals[tid][j];}
   printf("SHORTK_PMU KC=%d rep=%d threads=%d scaled=%d enabled=%lu running=%lu",KC,rep,nt,scaled,sums[1],sums[2]);
   for(int j=0;j<6;j++)printf(" %s=%lu",pmu_names[j],sums[j+3]);printf("\n");fflush(stdout);
  }
  pmu_on=0;
 }
 free(A);free(B);free(C);free(out);return 0;
}
'''
(p/'rns-shortk-pmu.c').write_text(s)
s=(p/'hybrid24-replicas.c').read_text();idx=s.index('static double val');s=s[:idx]+helper+s[idx:]
s=s.replace(' int bad=0;', ' int bad=0;pmu_init();',1)
s=s.replace('    sme_start();uint64_t t=cyc();','    pmu_on=1;pmu_begin(0);sme_start();uint64_t t=cyc();',1)
s=s.replace('uint64_t dt=cyc()-t;sme_stop();','uint64_t dt=cyc()-t;sme_stop();pmu_end(0);',1)
s=s.replace('    double ns=(double)dt', '    printf("H24_PMU variant=%d rep=%d nr=%lu enabled=%lu running=%lu",v,rep,pmu_vals[0][0],pmu_vals[0][1],pmu_vals[0][2]);for(int j=0;j<6;j++)printf(" %s=%lu",pmu_names[j],pmu_vals[0][j+3]);printf("\\n");\n    double ns=(double)dt',1)
(p/'hybrid24-pmu.c').write_text(s)
