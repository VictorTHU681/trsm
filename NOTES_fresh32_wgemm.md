# W-GEMM 路线终局（fresh32，2026-09-14）——实现完备、正确性全证、结构性证伪

**路线**：TC1（m≤1024 owner）求解改写 X = L⁻¹B 纯 GEMM。−W 写入 pp/diag（布局逐字节兼容，FP64 内核零改动）、
cbuf 存 RHS、解累加 wacc、尾块先行 dstore 直写、对角占优准入（Varah 界）+ 前向回退、RHS staging 解原地写回冲突、
64×64 块粒度右视递推（SME kloop 乘积 + epi_set/epi_sub 消费 za_dump 交错布局）、TRSM_WGEMM_NW 工人流水。

**正确性（全绿）**：自然准入 6 形状 + 强制 owner 4 形状 + margin 回退 + WGEMM=0 对照 + gen ×3 配置（wg2/wg3/wg5/wg6 四轮）；
W 链逐元素自检（TRSM_WGEMM_CHECK 标量参考）slot/pp/diag 全 ~1e-17；W 路径解误差 2.2~5e-16 不劣于前向。
三个实现坑全部定位修复：空区间线程 staging 链断裂（等界连跑寻址）、sc 层 J=K 需对角槽（fin 补写 +W_KK）、
za_dump scratch 为 tile 交错布局（消费改用生产级 epi_set_std/epi_sub_std，原语隔离测试实锤 got(off32)=want(8,0)）。

**性能（cn20483/81/86，官方 bench）**：v1 3.42（prep 1.33 纯C链无流水 + 尾块 scratch 往返）→ v2 3.36（SVE 乘积+NW10：prep 0.37 但 wait 0.58，32级×2相同步墙）
→ v3 2.91-2.97（64块粒度：prep 0.27 + wait 0.33 + 尾块冷读超额 0.22）vs off 1.88-1.90。NW 10/20/38 = 2.93/3.04/3.17（少工人+重叠更优）。

**结构性封顶（定价修正，勿再投入）**：原 +5~8% 估计误用 TC2 风格对角（56% 效率）作基线；TC1-owner 对角本就是
kloop_plain_nt+dstore 直写（0.217 ms，与 W 尾块同内核同工作量）→ 尾块最优=持平；khead 等待实测仅 0.002 ms；
尾部偏斜可回收 ≤0.04（SHAPEP 已收大半）；W 链即使完美流水 ≥0.02~0.1 ms 且合规禁止跨调用复用（L 不变也必须重算）。
净上限 ≈ ±0%，现实 −5% 以下。唯一残余价值：任务全独立的结构对共租户扰动可能更钝感（未验证，不足以立项）。
代码留 fresh32（默认关，TRSM_WGEMM=1 opt-in），正确性基建（wchk、原语测试、staging 协议）可复用。
