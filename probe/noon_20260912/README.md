# 2026-09-12 午间隔离原型

这一目录保存未合入生产的性能研究。实验固定基于 v47；归档前主线已新增队友的 v48，见研究记录。结果和限制见 [研究记录](../research_20260912_noon.md)，逐轮原始输出与统一 CSV 在 [logs/noon_20260912](../../logs/noon_20260912/)。

在仓库内运行：

```bash
python3 probe/noon_20260912/prepare.py
```

脚本从 Git 中提取固定的 `245bb36:trsm.c`，检查 SHA256，再生成各个研究源文件。生成物被本目录 `.gitignore` 排除，避免将实验当作生产实现。重新生成的十二个源文件已与实际上机文件逐字节比较一致；后来加入的 ANT 生成器也单独比较。

在已分配的 ARM 计算作业中加载 HPCKit GCC，再进入本目录编译，例如：

```bash
gcc -O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp hybrid24-duplex.c -o hybrid24-duplex -lm
OMP_NUM_THREADS=1 numactl -N 1 --preferred=17 taskset -c 40 ./hybrid24-duplex
```

这些内核探针默认使用一个核；正式 TRSM 对照使用 38 线程、NUMA1。不要在登录节点跑性能测试。

| 源文件 | 测量内容 |
|---|---|
| `hybrid-peak.c` | SME 与 SVE 纯寄存器并发，额外 1/2/4 行的算术容量 |
| `hybrid24.c`、`hybrid24-replicas.c` | 24×24 分解：8 个 ZA tile + SVE 8×8；运行时排列和预排副本比较 |
| `hybrid24-memory.c` | 包含 C 累加、驻留/单流/双流、KC32/64/128 |
| `hybrid24-duplex.c` | 两组 ZA 交替计算和回写，每组另算半个 SVE tile |
| `hybrid24-duplex-acc2.c` | 两组 SVE 累加器，检验依赖链假设 |
| `hybrid24-duplex-ant.c` | `-DH24_ANT=0/1/2`：普通 / 后半组非临时 / 两组非临时 A 读取 |
| `rns-shortk.c` | 13 模数 INT8、32×32 tile、KC64..2048、真实分段累加和 CRT |
| `rns-shortk-pmu.c`、`hybrid24-pmu.c` | 六事件 PMU，启用时间/运行时间均记录；不将仪器开销算作正式成绩 |
| `trsm-owner-import47.c` | TC1 在当前块计算时提前导入下一块；`TRSM_OIMPORT=0/1` 同二进制对照 |
| `sve-load-peak.c` | 独立 L1 load 与 SME/SVE 并发上限 |

`trsm-owner-import47.c` 是完整的实验性算子，可与仓库原始 `bench_trsm.c` 链接。其四轮正式对照回退，不能用于提交。RNS 只验证点积更新，输入编码不在计时内，没有完整 TRSM 的误差守卫或回退路径。

`python3 probe/noon_20260912/analyze.py` 从已提交原始日志重新生成 `measurements.csv`，保留所有样本。`plot.py` 生成阶段计时图。
