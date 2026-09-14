#define _GNU_SOURCE
#include <stdio.h>
#include <sys/prctl.h>
#include <sys/auxv.h>
#include <errno.h>
#include <arm_sve.h>
#ifndef PR_SME_SET_VL
#define PR_SME_SET_VL 63
#define PR_SME_GET_VL 64
#endif
int main(void) {
    long sve_old=prctl(PR_SVE_GET_VL,0,0,0,0),sme_old=prctl(PR_SME_GET_VL,0,0,0,0);
    errno=0;long sve_max=prctl(PR_SVE_SET_VL,256,0,0,0);int e1=errno;
    errno=0;long sme_max=prctl(PR_SME_SET_VL,256,0,0,0);int e2=errno;
    unsigned long normal,streaming;
    asm volatile("cntb %0":"=r"(normal));
    asm volatile("smstart\n\tcntb %0\n\tsmstop":"=r"(streaming)::"memory");
    printf("VECTOR_LIMITS sve_old=%ld sme_old=%ld sve_max=%ld sve_errno=%d sme_max=%ld sme_errno=%d normal_bytes=%lu streaming_bytes=%lu\n",sve_old,sme_old,sve_max,e1,sme_max,e2,normal,streaming);
    if(sve_old>=0)prctl(PR_SVE_SET_VL,sve_old,0,0,0);
    if(sme_old>=0)prctl(PR_SME_SET_VL,sme_old,0,0,0);
    return sve_max<0||sme_max<0;
}
