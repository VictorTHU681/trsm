# TRSM on Kunpeng 72F5 (SME) — 鲲鹏 HPC 挑战赛 S2 TRSM 赛题

手写 SME/SVE 的 `l_trsm(m, n, L, lda, B, ldb)`：解 L·X = B（L 行主序下三角，B/X 行主序，原地），
支持任意合法 m/n/lda/ldb，无 SME 时回退通用实现。只改 `trsm.c`。

## 成绩（判题口径 `OMP_NUM_THREADS=38 numactl -N 1 ./trsm_test m n 1`，cn2048x，v15，两轮）

| 用例 | 时间 | GFLOPS | 占 FP64 峰值 7.54 TFLOPS | 占带宽下限 | vs KML 26.1 dtrsm（绑核 / 判题口径） |
|---|---|---|---|---|---|
| 512 × 19968 | 3.1–3.3 ms | 1575–1700 | 21–23% | 55% | 1.2× / 50× |
| 2432 × 17024 | 33.8 ms | 2980 | 40% | 78% | 1.15× / 39× |
| 17024 × 512 | 47.4 ms | 3125 | 41% | 74% | 12× / 200× |

三用例数值误差 ≤ 1.5e-15（阈值 1e-12）；随机形状泛化测试（probe/gen_test.c，含非对齐/奇数 lda/ldb）全部通过。

"带宽下限"：本机 38 核聚合读带宽封在 ~305 GB/s（与 HBM/DDR/远端 HBM 组合无关，见 NOTES §1），
每核 L2 768 KB、无共享 L3 → C 块驻留阻塞的最小流量给出 case2/3 的上限约 50–56% 峰值，case1 约 40%。
80%/85% 峰值的目标在这台机器上不可达；详细推导、迭代记录和坑见 [NOTES.md](NOTES.md)。

## 编译运行

```bash
bash run.sh          # 加载 HPCKit 环境、编译（SME 指令集）、按判题口径跑三用例（自动选最近的 HBM 节点 --preferred）
# 或按题面：
gcc -O3 -march=armv9-a+sme+sme-f64f64+sve2 bench_trsm.c trsm.c -o trsm_test -lm -lkblas -fopenmp
OMP_NUM_THREADS=38 numactl -N 1 ./trsm_test 2432 17024 1
```

调参环境变量（默认已按三用例选好）：`TRSM_MB/NC/KC`（块大小，默认 192/128/64）、`TRSM_ANT`（GEMM A 条 ldnt1d）、
`TRSM_SWHOST`（X 回写托管）、`TRSM_RESERVE`（对角阶段预占下一任务）、`TRSM_SPLITF`（K 分裂）、`TRSM_PROFILE=2`（分相计时）。

## 结构

- `trsm.c` — 算子：16×32 FMOPA 微内核（内联 asm）、托管回写 K 循环、对角块 `[-Linv·L | Linv]` 预处理、任务调度、L 打包、NUMA 放置。
- `bench_trsm.c` — 题目给的基准（未改）。`run.sh` — 判题方式脚本。
- `probe/` — 探针与实验：`peak.c`（FMOPA 峰值 / 带宽）、`bw_nodes.c`（多节点 / 共享读带宽）、`kbench.c`（单线程内核微基准）、
  `gen_test.c`（随机形状正确性）、`trsm_kblas.c`（KML 基线）、`*_job.sh`（Donau 作业）、`submit.sh`（挑空闲节点提交）。
- `logs/` — 各实验作业输出。
- `NOTES.md` — 实测数据、roofline、设计、迭代表、损耗分析、教训。
