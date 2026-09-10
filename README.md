# 鲲鹏高性能计算全球挑战赛（S2赛季）TRSM 优化赛题

## 编译
``` shell
gcc -O3 bench_trsm.c trsm.c -o trsm_test -lm -lkblas -fopenmp
```

## 运行
本次三个测试用例，运行方法如下：
``` shell
OMP_NUM_THREADS=38 numactl -N 1 ./trsm_test 512 19968 1 #参数依次是 m n test_runs表示重复测试次数，使性能稳定即可
OMP_NUM_THREADS=38 numactl -N 1 ./trsm_test 2432 17024 1
OMP_NUM_THREADS=38 numactl -N 1 ./trsm_test 17024 512 1
```

## 调优
修改trsm.c源文件的l_trsm函数，优化性能。

校验结果无误，按性能（GFLOPS）排名。