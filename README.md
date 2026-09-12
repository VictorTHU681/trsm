# TRSM on Kunpeng 72F5 (SME) — 鲲鹏 HPC 挑战赛 S2 TRSM 赛题

手写 SME/SVE 的 `l_trsm(m, n, L, lda, B, ldb)`：解 L·X = B（L 行主序下三角，B/X 行主序，原地），
支持任意合法 m/n/lda/ldb，无 SME 时回退通用实现。只改 `trsm.c`。

## 当前成绩

v47 在 cn20486 的原始 bench 上与 v46 交替测试六轮，每进程计时 11 次，均为 38 线程、NUMA1：

| 用例 | v46 中位 GFLOPS | v47 中位 GFLOPS | 配对性能变化 | 最大误差 |
|---|---:|---:|---:|---:|
| 512 × 19968 | 2777.7652 | 2782.9089 | +0.16%（持平） | 3.33e-16 |
| 2432 × 17024 | 4173.1966 | 4209.0075 | +0.71% | 7.77e-16 |
| 17024 × 512 | 4131.9069 | 4160.9955 | +0.56% | 1.50e-15 |

配对变化为同轮性能比值的几何平均；保留全部轮次，见 [v47 原始数据](logs/v47_formal.csv)。这是本地集群成绩，不是线上评测。

v47 官方 bench 正式中位数（cn20484，TC1×4 / TC2×8 / TC3×4，题面原样命令）：**1.905 ms / 2749、23.87 ms / 4217、35.62 ms / 4167**（范围 1.89–1.94 / 23.70–24.00 / 35.59–35.78），合计 ≈ 11133。线上（2 GHz）v41 包为 2862 / 4821 / 4665。
v47 的固定 KC64 内核用立即偏移替代部分地址加法；保留原内核，以 `TRSM_ABHOIST=0` 在同一二进制中对照。TC1 owner 路径保持原实现，v45 大页及 v46 忙核探测均保留。
64 组泛化、192 组独立已知解（含同址切换新旧内核）和 12 次正式尺寸换输入检查均通过；最终源码与完整复验候选的 ARM .text 一致，新内核另通过保护页逐元素精确检查及 A4NT 开关编译验证。

此前 v44 的 TC1 复验有慢轮，完整范围为 2048.9521–2812.7300 GFLOPS；TC2/3 为 4157.2915–4198.1107 / 4132.0599–4142.5656。保留全部轮次，不剔除低值，见 [原始逐轮数据](logs/v44_inline_formal.csv)。此前线上 v41 为 2862 / 4821 / 4665 GFLOPS，和本机测试分开记录。

v44 只为四个 KC64 主内核添加强制内联，减少调用和寄存器保存开销。全部 FP64 算术，不依赖 bench 生成器的数值特性（v32 已移除 Q32）。
64 组泛化、192 组独立 53 位输入精度测试（含专用路径断言、极大/极小缩放、同地址换输入、非对齐和填充保护）、12 次正式尺寸换输入均通过；无 SME 的通用编译通过。
case1 走面板驻留模式（`TRSM_OWNER`，m≤1024 且面板数≥2×线程数时自动），case2/3 走两组 ZA bank 流水的专用内核。
设计、迭代数据、判题机时钟与效率口径见 [NOTES.md](NOTES.md)。

## 编译运行

```bash
bash run.sh          # 加载 HPCKit 环境、编译（SME 指令集）、按判题口径跑三用例（自动选最近的 HBM 节点 --preferred）
# 或按题面：
gcc -O3 -march=armv9-a+sme+sme-f64f64+sve2 bench_trsm.c trsm.c -o trsm_test -lm -lkblas -fopenmp
OMP_NUM_THREADS=38 numactl -N 1 ./trsm_test 2432 17024 1
```

快速迭代（不跑官方 bench 的 cblas_dgemm 参考解，改抽样残差校验；整个三用例 ~20 s，加变体各 ~15 s）：

```bash
bash probe/fast.sh                                  # 提交到空闲节点、等待、打印三用例 min/avg
bash probe/fast.sh "TRSM_KC=96" "TRSM_MB=256 TRSM_KC=128"   # 附带变体（每个参数一组环境变量）
OMP_NUM_THREADS=38 numactl -N 1 ./fastbench -r 5 2432 17024  # 在计算节点上直接跑（-r 轮数，-s 抽样行数，0 关校验）
```

调参环境变量（默认已按三用例选好）：`TRSM_MB/NC/KC`（块大小，默认 192/128/64；大宽矩阵 256/128/64；面板驻留模式 64/96/64）、`TRSM_OWNER`（-1 自动 / 0 关 / 1 强制面板驻留，见 NOTES §9）、`TRSM_ANT`（GEMM A 条 ldnt1d）、`TRSM_XNODE/XPNODE/LNODE0`（HBM 放置）、
`TRSM_SWHOST`（X 回写托管）、`TRSM_RESERVE`（对角阶段预占下一任务）、`TRSM_SPLITF`（K 分裂）、`TRSM_PROFILE=2`（分相计时）。

## 结构

- `trsm.c` — 算子：16×32 FMOPA 微内核（内联 asm）、托管回写 K 循环、对角块 `[-Linv·L | Linv]` 预处理、任务调度、L 打包、NUMA 放置。
- `bench_trsm.c` — 题目给的基准（未改）。`run.sh` — 判题方式脚本。
- `probe/` — 探针与实验：`fastbench.c` + `fast.sh`（快速基准）、`peak.c`（FMOPA 峰值 / 带宽）、`bw_nodes.c`（多节点 / 共享读带宽）、
  `kbench.c`（单线程内核微基准）、`gen_test.c`（随机形状正确性）、`trsm_kblas.c`（KML 基线）、`*_job.sh`（Donau 作业）、`submit.sh`（挑空闲节点提交，候选缓存）。
- `logs/` — 各实验作业输出。
- `NOTES.md` — 实测数据、roofline、设计、迭代表、损耗分析、教训。
