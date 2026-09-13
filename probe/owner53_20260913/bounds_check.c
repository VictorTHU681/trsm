#include "trsm-owner53.c"
int main(void)
{
    const int rows[]={1,2,3,4,5,7,8,11,16};
    const int panels[]={1,2,3,7,38,39,104,208,209,511};
    const int threads[]={1,5,19,38,76};
    const int weights[]={0,160,192,300};
    long cases=0,tasks=0,bounds=0;
    setenv("TRSM_KHEADW","0",1);
    for(int r=0;r<9;r++)for(int p=0;p<10;p++)for(int th=0;th<5;th++)for(int w=0;w<4;w++){
        char weight[32];snprintf(weight,sizeof(weight),"%d",weights[w]);setenv("TRSM_OWNER_C0",weight,1);
        ctx_t c={0};
        c.mb=64;c.nc=96;c.kc=64;c.nrb=rows[r];c.npan=panels[p];c.ncp=96;
        c.m=c.nrb*c.mb;c.n=c.npan*c.nc;c.owner=1;c.special=5;c.pf2=1;
        c.S_of=malloc(c.nrb*sizeof(int));for(int i=0;i<c.nrb;i++)c.S_of[i]=4;
        build_tasks(&c,threads[th]);
        long* old=c.owner_bound;
        for(long q=0;q<c.ntask_items;q++){
            task_t x=owner_task_at(&c,q),y=c.tasks[q];
            if(x.kind!=y.kind||x.g!=y.g||x.i!=y.i||x.p!=y.p||x.seg!=y.seg||x.nseg!=y.nseg||x.ch0!=y.ch0||x.ch1!=y.ch1||x.part_off!=y.part_off){
                printf("OWNER_TASK FAIL rows=%ld panels=%ld q=%ld\n",c.nrb,c.npan,q);return 1;
            }
            tasks++;
        }
        c.implicit_owner=1;build_tasks(&c,threads[th]);
        for(int t=0;t<=threads[th];t++){
            if(old[t]!=c.owner_bound[t]){
                printf("OWNER_BOUND FAIL rows=%ld panels=%ld nth=%d weight=%d t=%d old=%ld new=%ld\n",c.nrb,c.npan,threads[th],weights[w],t,old[t],c.owner_bound[t]);return 1;
            }
            bounds++;
        }
        free(old);free(c.owner_bound);free(c.S_of);cases++;
    }
    printf("OWNER_BOUNDS PASS cases=%ld tasks=%ld boundaries=%ld\n",cases,tasks,bounds);
    return 0;
}
