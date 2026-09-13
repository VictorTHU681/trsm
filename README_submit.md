# TRSM 提交包

`l_trsm(m, n, L, lda, B, ldb)`：解 L·X = B（L 为行主序 m×m 下三角矩阵，B/X 行主序 m×n，原地覆盖）。
FP64 SME(FMOPA) 实现，支持任意合法维度与步长；无 SME 硬件或编译器不支持时自动回退通用实现。

## 文件

- `trsm.c` — 算子实现。
- `bench_trsm.c` — 题目提供的基准程序（未改动）。
- `run.sh` — 编译并按判题方式运行三组用例。

## 运行

```bash
bash run.sh
```

run.sh 自动加载 HPCKit 编译环境（兼容题面 module 路径与本集群 setvars.sh 两种布局），
以 `-march=armv9-a+sme+sme-f64f64+sve2` 编译；内存放置用 `numactl -N 1 --preferred=<就近纯内存节点>`
（按 NUMA 距离表探测，探测不到时自动省略该参数）。也可按题面命令直接编译运行：

```bash
gcc -O3 -march=armv9-a+sme+sme-f64f64+sve2 bench_trsm.c trsm.c -o trsm_test -lm -lkblas -fopenmp
OMP_NUM_THREADS=38 numactl -N 1 ./trsm_test 2432 17024 1
```

## 实现要点

16×32 FMOPA 微内核（内联汇编、双 ZA bank 流水）、行块×列面板任务并行 + 大任务 K 分裂、
对角块 [-L⁻¹L | L⁻¹] 预处理（求解阶段无除法）、软件预取走线器、线程显式绑核、
自有缓冲的 NUMA 放置（mbind MPOL_PREFERRED，仅作用于自身分配的内存）。
每次调用完整重建全部打包与预处理，无跨调用缓存，无按用例尺寸的特判。
