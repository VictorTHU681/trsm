// 碎片制造器：把指定 NUMA 节点的空闲内存打成 <2MiB 的碎块并钉住一半，
// 复现判题机"退化态"（THP 分配失败 → bench 缓冲退 4 KB 页）。
// 用法：./fragger <node> <MB> [sleep_s]
//   在 <node> 上以 4 KB 页触碰 <MB> 兆，随后每 2 MiB 对齐块内释放前 1 MiB
//  （空闲表只剩 order<9 的块），保持驻留直到收到 SIGTERM 或睡满 sleep_s。
// 度量：前后打印 /proc/buddyinfo 中该节点 order>=9 的空闲块数。
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/syscall.h>

#ifndef MPOL_PREFERRED
#define MPOL_PREFERRED 1
#endif
#ifndef MADV_NOHUGEPAGE
#define MADV_NOHUGEPAGE 15
#endif

static void buddy(int node)
{
    FILE* f = fopen("/proc/buddyinfo", "r");
    if (!f) return;
    char line[512];
    char tag[32];
    snprintf(tag, sizeof tag, "Node %d,", node);
    while (fgets(line, sizeof line, f))
        if (strstr(line, tag)) fputs(line, stderr);
    fclose(f);
}

int main(int argc, char** argv)
{
    if (argc < 3) { fprintf(stderr, "用法：%s <node> <MB> [sleep_s] [keep_per_2MB] [poison_pct]\n", argv[0]); return 2; }
    int node = atoi(argv[1]);
    size_t mb = (size_t)atol(argv[2]);
    long secs = argc > 3 ? atol(argv[3]) : 3600;
    int keep = argc > 4 ? atoi(argv[4]) : 16;      // 每 2MB 块钉住的 4KB 页数（散布），容量近乎全部归还
    int pct  = argc > 5 ? atoi(argv[5]) : 100;     // 毒化的块比例（其余块整块释放）——覆盖率旋钮
    size_t bytes = mb << 20;
    fprintf(stderr, "[fragger] before:\n"); buddy(node);
    void* p = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) { perror("mmap"); return 1; }
    unsigned long mask[16] = {0};
    mask[node / 64] |= 1UL << (node % 64);
    if (syscall(SYS_mbind, p, bytes, MPOL_PREFERRED, mask, (unsigned long)(sizeof(mask) * 8), 0))
        perror("mbind");
    madvise(p, bytes, MADV_NOHUGEPAGE);   // 强制 4 KB 页
    // 触碰全部页（占住节点内存）
    for (size_t o = 0; o < bytes; o += 4096) ((volatile char*)p)[o] = 1;
    // 毒化：pct% 的 2MiB 块里散布钉住 keep 个 4KB 页（其余释放）→ 阻止这些块合并出 order-9；
    // 其余块整块释放。容量几乎全部归还，只毒化连续性。
    const size_t two = (size_t)2 << 20;
    uintptr_t a0 = ((uintptr_t)p + two - 1) & ~(two - 1);
    size_t freed = 0, chunk_i = 0;
    for (uintptr_t a = a0; a + two <= (uintptr_t)p + bytes; a += two, chunk_i++) {
        if ((int)(chunk_i % 100) < pct && keep > 0) {
            size_t stride = two / keep;
            for (int s = 0; s < keep; s++) {
                uintptr_t seg = a + (size_t)s * stride;
                size_t len = stride - 4096;      // 每段留末尾 1 页
                if (len) { madvise((void*)seg, len, MADV_DONTNEED); freed += len; }
            }
        } else {
            madvise((void*)a, two, MADV_DONTNEED);
            freed += two;
        }
    }
    fprintf(stderr, "[fragger] node=%d 触碰 %zuMB，keep=%d/2MB pct=%d%%，释放 %zuMB，钉住 %zuMB；after:\n",
            node, mb, keep, pct, freed >> 20, (bytes - freed) >> 20);
    buddy(node);
    fflush(stderr);
    sleep((unsigned)secs);
    return 0;
}
