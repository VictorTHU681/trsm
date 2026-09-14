# 未完整下载日志的 SSH 观测

这不是完整原始日志。以下仅记录 2026-09-14 SSH 工具已经返回的尾部内容，便于网络恢复后核对。
完整文件仍应从 `/home/share/niujiayin/trsm-breakthrough56-20260914/results/` 取回。

489 已显示 `RECURSIVE_RIGHT_DONE`。最后一个配置为 TC3，NB1024、NC128、MB64、jouter=1：
原始 bench 143.88 ms / 1031.3125 GFLOPS，最大误差 1.52e-14，PASS。
三个暖 profile 总时间分别为 142.465133、139.918500、148.803360 ms。
不能据此声称其他配置的完整数据已归档。

已观察到的源文件哈希：

```text
4e3ffc19bc611432f6dd0be25135edab29738a63cfea173854d8a247d1506e8f  tiled-recursive.c
9f50165b3dd3108b1ec60c49155412ee09a60830266282bd821611e504857aab  block_inverse.h
d1c9e19dc0a11550ca6aba3bb53449d20446fd96c1a462c7eb6ee7fe8b2d4d67  right-update.h
```

490 已显示 `PLATFORM_LIMITS_DONE`：

```text
VECTOR_LIMITS sve_old=64 sme_old=64 sve_max=64 sve_errno=0 sme_max=64 sme_errno=0 normal_bytes=64 streaming_bytes=64
74cfe00378131c307de8b8bf4b78802276cb2ea2966e4dc0e15de18f92b0bd63  vector_limits.c
```

`lscpu` 显示 608 个 CPU、每核 1 线程、NUMA1 为 CPU38–75，L1D 总计 19 MiB/608 份、L2 总计 456 MiB/608 份。
这些总数不代表测试使用了 608 核；完整求解器仍为 38 线程。
