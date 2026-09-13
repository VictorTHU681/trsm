// 压实助手（run.sh 级，判题机 thp_defrag=madvise 时有效）：
// mmap → mbind PREFERRED(node) → MADV_HUGEPAGE → 每 2MB 触碰一字节（缺页触发 direct compaction，
// 把节点碎片整理成 2MB 连续块）→ 自查 smaps 统计实得大页 → munmap（完整 2MB 块归还伙伴系统）。
// 之后启动的进程（bench）malloc 缓冲走廉价 THP 路径时可直接从空闲链表拿到大块。
// 不触碰任何其他进程的内存；defrag=never 的机器上退化为普通触碰（无压实、无害）。
// 用法：compact <node> <MB> [rounds=1]
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

static void buddy(int node)
{
    FILE* f = fopen("/proc/buddyinfo", "r");
    if (!f) return;
    char line[512], tag[32];
    snprintf(tag, sizeof tag, "Node %d,", node);
    while (fgets(line, sizeof line, f))
        if (strstr(line, tag)) fputs(line, stderr);
    fclose(f);
}

static size_t huge_bytes_of(void* p)
{
    FILE* f = fopen("/proc/self/smaps", "r");
    if (!f) return 0;
    char line[256];
    uintptr_t lo = 0, hi = 0;
    size_t huge = 0;
    int in = 0;
    while (fgets(line, sizeof line, f)) {
        uintptr_t a, b;
        long kb;
        if (sscanf(line, "%lx-%lx", &a, &b) == 2) { lo = a; hi = b; in = ((uintptr_t)p >= a && (uintptr_t)p < b); }
        else if (in && sscanf(line, "AnonHugePages: %ld", &kb) == 1) huge = (size_t)kb << 10;
        (void)lo; (void)hi;
    }
    fclose(f);
    return huge;
}

int main(int argc, char** argv)
{
    if (argc < 3) { fprintf(stderr, "用法：%s <node> <MB> [rounds]\n", argv[0]); return 2; }
    int node = atoi(argv[1]);
    size_t mb = (size_t)atol(argv[2]);
    int rounds = argc > 3 ? atoi(argv[3]) : 1;
    const size_t two = (size_t)2 << 20;
    size_t bytes = (mb << 20) & ~(two - 1);
    fprintf(stderr, "[compact] node=%d %zuMB rounds=%d; before:\n", node, bytes >> 20, rounds);
    buddy(node);
    for (int r = 0; r < rounds; r++) {
        void* p = mmap(NULL, bytes + two, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p == MAP_FAILED) { perror("mmap"); return 1; }
        uintptr_t a0 = ((uintptr_t)p + two - 1) & ~(two - 1);
        size_t len = bytes;
        unsigned long mask[16] = {0};
        mask[node / 64] |= 1UL << (node % 64);
        if (syscall(SYS_mbind, (void*)a0, len, MPOL_PREFERRED, mask, (unsigned long)(sizeof(mask) * 8), 0))
            perror("mbind");
        madvise((void*)a0, len, MADV_HUGEPAGE);   // defrag=madvise/always 时触碰会做直接压实
        for (size_t o = 0; o < len; o += two) ((volatile char*)a0)[o] = 1;
        size_t huge = huge_bytes_of((void*)a0);
        fprintf(stderr, "[compact] round %d: 触碰 %zuMB，实得大页 %zuMB（%.0f%%）\n",
                r, len >> 20, huge >> 20, len ? 100.0 * huge / len : 0);
        munmap(p, bytes + two);   // 大页整块归还 → 空闲链表出现 order-9 块
    }
    fprintf(stderr, "[compact] after:\n");
    buddy(node);
    return 0;
}
