#include "trsm-diagcompact.c"
typedef struct {void*base;size_t bytes;double*ptr;} guarded_t;
static guarded_t guard_alloc(size_t bytes) {
    size_t page=sysconf(_SC_PAGESIZE),data=(bytes+page-1)/page*page,total=data+2*page;
    void*base=mmap(NULL,total,PROT_NONE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    if(base==MAP_FAILED||mprotect((char*)base+page,data,PROT_READ|PROT_WRITE))abort();
    return (guarded_t){base,total,(double*)((char*)base+page+data-bytes)};
}
static uint64_t guard_rng=UINT64_C(0x13293dbd5723456);
static double guard_value(void){guard_rng^=guard_rng>>12;guard_rng^=guard_rng<<25;guard_rng^=guard_rng>>27;return ((guard_rng*UINT64_C(2685821657736338717))>>11)*0x1p-53-.5;}
int main(void){
    if(svcntd()!=8)return 2;
    guarded_t ag=guard_alloc(256*16*8),bg=guard_alloc(256*128*8),cg=guard_alloc(16*128*8),sg=guard_alloc(512*8);
    _Alignas(64) double result[16*128],dummy[8192];size_t checked=0;
    memset(dummy,0,sizeof(dummy));
    for(int scale=0;scale<3;scale++)for(int K=16;K<=256;K+=16)for(int epi=0;epi<2;epi++){
        if(epi&&K<64)continue;
        double*a=ag.ptr+(256-K)*16,*b=bg.ptr+(256-K)*128,*c=cg.ptr,*scratch=sg.ptr;
        double as=scale==1?0x1p400:scale==2?0x1p-400:1,bs=1/as;
        for(int k=0;k<K;k++){
            for(int r=0;r<16;r++)a[k*16+r]=guard_value()*as;
            for(int j=0;j<128;j++)b[k*128+j]=guard_value()*bs;
        }
        double original[512];
        for(int z=0;z<512;z++)scratch[z]=original[z]=guard_value();
        for(int z=0;z<16*128;z++)c[z]=result[z]=0x1.29p10;
        pf_t pf={(char*)dummy,16,0,(char*)dummy,(char*)dummy,1024,&g_sw_none};
        sme_start();sme_zero_za();
        if(!epi)kloop_diag_plain_compact(a,128,128,b,1024,K,32,&pf);
        else if(K==64)kloop_diag64_set_compact(a,128,128,b,1024,32,&pf,scratch,c,1024,32);
        else kloop_diag_set_compact(a,128,128,b,1024,K,32,&pf,scratch,c,1024,32);
        za_dump(scratch);epi_set_std(scratch,result,1024,32);sme_stop();
        for(int r=0;r<16;r++)for(int j=0;j<128;j++){
            double expected=0x1.29p10;
            if(j<32){expected=0;for(int k=0;k<K;k++)expected=fma(a[k*16+r],b[k*128+j],expected);}
            if(result[r*128+j]!=expected||!isfinite(result[r*128+j])){printf("COMPACT_GUARD FAIL K=%d epi=%d r=%d j=%d\n",K,epi,r,j);return 1;}
            double previous=epi&&j<32?original[(r%8)*64+(r/8)*32+j]:0x1.29p10;
            if(c[r*128+j]!=previous){puts("COMPACT_EPI_GUARD FAIL");return 1;}
            checked+=2;
        }
    }
    munmap(ag.base,ag.bytes);munmap(bg.base,bg.bytes);munmap(cg.base,cg.bytes);munmap(sg.base,sg.bytes);
    printf("COMPACT_GUARD comparisons=%zu PASS\n",checked);return 0;
}
