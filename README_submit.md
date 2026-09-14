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
以 `-march=armv9-a+sme+sme-f64f64+sve2 -mtune=tsv110 -flto` 编译（编译器不支持时自动降级为
基础 SME 编译，再不支持则回退通用实现）；CPU 绑定默认 node 7（不可用时自动回退 node 1），
内存放置用 `--preferred=<就近纯内存节点>`（按 NUMA 距离表探测，探测不到时自动省略该参数）。
也可按题面命令直接编译运行：

```bash
gcc -O3 -march=armv9-a+sme+sme-f64f64+sve2 bench_trsm.c trsm.c -o trsm_test -lm -lkblas -fopenmp
OMP_NUM_THREADS=38 numactl -N 1 ./trsm_test 2432 17024 1
```

## 实现要点

16×32 FMOPA 微内核（内联汇编、双 ZA bank 流水）、行块×列面板任务并行 + 大任务 K 分裂、
对角块 [-L⁻¹L | L⁻¹] 预处理（求解阶段无除法）、软件预取走线器、线程显式绑核、
自有缓冲的 NUMA 放置（mbind MPOL_PREFERRED，仅作用于自身分配的内存）。
面板边界按 X 的 cache 行对齐折叠（X 基址错位且 n 为面板宽整数倍时，首面板收编尾部错位列，
其余面板装载/回写全对齐；通用结构谓词准入，无用例特判）。窄面板用例的精简对角内核
（固定步长、去除空托管回写与越界预取，FP64 运算次序与结果不变）。
首次（不计时的校验）调用把 X 缓冲物理页重铺为 2 MiB 透明大页（拷出→MADV_DONTNEED→
MADV_HUGEPAGE→拷回，虚拟地址与内容逐字节不变；已按报备获官方批准，`TRSM_REHUGE=0` 可关闭）。
每次调用完整重建全部打包与预处理，无跨调用缓存，无按用例尺寸的特判。
