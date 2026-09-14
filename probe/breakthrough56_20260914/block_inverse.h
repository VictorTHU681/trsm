// Local recursive triangular inversion using SME products. Research prototype.
static void inverse_product(const double*A,long la,const double*B,long lb,double*C,long lc,
                            int R,int N,int K,double*ap,double*bp,int negate) {
    sme_start();
    for(int r=0;r<R;r+=16)pack_strip(A,la,R,r,0,K,ap+(size_t)r*K);
    for(int j=0;j<N;j+=32)for(int k=0;k<K;k++)inv_copy32(B+(size_t)k*lb+j,bp+(size_t)j*K+k*32);
    for(int r=0;r<R;r+=16)for(int j=0;j<N;j+=32) {
        sme_zero_za();inv_compact(ap+(size_t)r*K,bp+(size_t)j*K,K);
        za_store_direct(C+(size_t)r*lc+j,lc*8,32);
    }
    if(negate) {
        svbool_t pg=svptrue_b64();
        for(int r=0;r<R;r++)for(int j=0;j<N;j+=8)svst1_f64(pg,C+(size_t)r*lc+j,svneg_f64_x(pg,svld1_f64(pg,C+(size_t)r*lc+j)));
    }
    sme_stop();
}
static void invert_block(const double*L,long lda,int K,double*inv,int nb,double*ap,double*bp,double*tmp) {
    for(int i=0;i<K;i++)for(int j=0;j<K;j++)inv[(size_t)i*nb+j]=0;
    for(int base=0;base<K;base+=32)for(int i=base;i<base+32;i++) {
        double*row=inv+(size_t)i*nb;
        double di=L[(size_t)i*lda+i];row[i]=1.0/di;
        for(int k=base;k<i;k++) {
            double v=-L[(size_t)i*lda+k]/di;
            for(int j=base;j<=k;j+=8) {
                svbool_t pg=svwhilelt_b64((uint64_t)j,(uint64_t)(k+1));
                svst1_f64(pg,row+j,svmla_n_f64_x(pg,svld1_f64(pg,row+j),svld1_f64(pg,inv+(size_t)k*nb+j),v));
            }
        }
    }
    for(int width=64;width<2*K;width*=2)for(int base=0;base<K;base+=width) {
        int h=width/2,d=K-base-h;if(d<=0)continue;if(d>h)d=h;
        inverse_product(L+(size_t)(base+h)*lda+base,lda,inv+(size_t)base*nb+base,nb,tmp,h,d,h,h,ap,bp,0);
        inverse_product(inv+(size_t)(base+h)*nb+base+h,nb,tmp,h,inv+(size_t)(base+h)*nb+base,nb,d,h,d,ap,bp,1);
    }
}
