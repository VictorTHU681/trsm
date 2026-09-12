# v50 的实验与复验

固定基线 `2184f49:trsm.c`（v48，SHA256 `04c6e356a287bc6afe460500ed177a7fd68ff847d217868481e4be77191a346f`）。
生产版本为 v50 / `TRSM_OWNER_CURSOR`；402 脚本及其文件沿用实验当时的 `v49-final` 名称，与队友已撤回的 PCACHE 无关。

```bash
python3 probe/v50_20260912/prepare.py
python3 probe/v50_20260912/analyze.py
```

prepare 从固定提交提取生产源码，展开原有 KC64 宏，再生成 A/B 地址、预取地址、对角内核等候选。
其中十份候选、审计和边界检查文件已与实际 ARM 测试文件逐字节比较一致。生成文件不会改动生产 `trsm.c`。
`trsm-v45-codex.c` 是供旧生成器读取的文件名，此处内容仍是固定 v48；原 owner 宏和内核未变。

395–402 脚本记录当时执行的命令，在计算节点加载 HPCKit 26.1.RC1 gcc/kblas 后顺序运行。
使用 38 线程、NUMA1，独立边界探针固定单核 CPU40；不要在登录节点运行，也不要与其他计算重叠。
脚本中的 `ref-removal.o` 为通用参考解，复现时可从固定基线重新构建：

```bash
cd probe/v50_20260912
gcc -O3 -fopenmp -DTRSM_NO_SME_ASM -Dl_trsm=l_trsm_ref -c trsm-v48-synced.c -o ref-removal.o
export OMP_NUM_THREADS=38 OMP_PROC_BIND=close OMP_PLACES=cores
export OMP_DYNAMIC=false OMP_MAX_ACTIVE_LEVELS=1 OMP_WAIT_POLICY=active
```

严格精度检查另有 192 组独立 53 位已知解和 12 次正式尺寸同址换输入，不只依靠通用参考实现。
原始 bench 可能在打印 FAIL 后仍以 0 退出，必须同时检查日志中的校验字段。

结果、统计定义和未采用的候选见 [研究记录](../research_v50_20260912.md)。
