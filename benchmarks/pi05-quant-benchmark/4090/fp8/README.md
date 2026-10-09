# PI0.5 · 4090 · FP8

当前 [config.json](config.json) 使用 **B=4、前缀 native W8A8、动作专家 BF16**。
126 个前缀投影使用 E4M3 权重/激活与 tensorwise FP32 scale；视觉编码与辅助计算
保留 Inductor，前缀及完整十步去噪均使用 CUDA Graph，attention 使用 Triton。
动态激活量化使用融合归约与转换，未保存整张 FP32 激活副本。

[实验方法与环境](../../README.md)：LIBERO-10，1,600 条观测，两个 RGB、状态与指令，
seed=42，完整十步去噪和 50×7 物理动作块。B=4 完整预热 400 次 batch 调用。
E2E 包括 CPU 输入预处理、传输、全部模型计算和 CPU 动作后处理；排除加载、
图捕获、磁盘解码、HTTP 和仿真。吞吐按实际观测数计算，不使用 action slots/s。

2026-10-10 同机完整单卡先导结果；Torch 2.13.0+cu129、Transformers 5.3.0、
LeRobot 0.5.1、Triton 3.7.1，4090 24,564 MiB，450 W，四个 CPU 线程。
前缀 FP8 的 B=1 测量与另一张卡的教师数据采集重叠；不能替代隔离复测。

| 配置 | Batch | E2E ms/观测 | obs/s/卡 | 峰值 allocated GiB |
| --- | ---: | ---: | ---: | ---: |
| BF16，原生优化与 Inductor | 1 | 42.83 | 23.35 | 10.80 |
| 全投影 native FP8，channelwise，Inductor 关闭 | 1 | 66.53 | 15.03 | 5.88 |
| 前缀 FP8，专家 BF16，Inductor 开启 | 1 | 36.19 | 27.63 | 6.54 |
| BF16，原生优化与 Inductor | 4 | 29.74 | 33.63 | 11.16 |
| 前缀 FP8，专家 BF16，Inductor 开启 | 4 | 24.02 | 41.63 | 6.91 |

同 batch 对比，B=4 的 E2E 加速约 1.24 倍、峰值显存下降约 38%。大前缀矩阵
获得 FP8 GEMM 收益；动作专家的小矩阵动态量化开销较大，因此保留 BF16。
B=4 正式测量前后图捕获计数均为 4，未在计时区间新增捕获。

量化有损。与 B=4 BF16 同样本/噪声/步数比较，全部物理动作的 MAE 为 0.004145，
RMSE 为 0.033480，最大绝对误差为 2.00924；夹爪维 RMSE 为 0.084238。
这些维度单位不同，误差不能替代 LIBERO 成功率评估。本次没有运行闭环成功率。
B=1 的对应 MAE/RMSE 为 0.004124/0.034547。

双卡隔离复测已完成：两张卡各回放同样的 1,600 条观测，预热后由共同 barrier
启动，期间没有训练或回归测试。这里只报告每卡效率，不把重复数据当作更多独立样本。

| B=4 双实例 | GPU 0 obs/s | GPU 1 obs/s | 平均每卡 obs/s | E2E ms/观测 | 峰值 allocated GiB/卡 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BF16 | 33.54 | 33.64 | 33.59 | 29.77 | 11.16 |
| 前缀 FP8 / 专家 BF16 | 41.68 | 41.50 | **41.59** | **24.04** | **6.91** |

配对 E2E 加速 **1.238 倍**、峰值 allocated 下降 **38.14%**；FP8 reserved 为
8.91 GiB/卡。每个进程正式测量前后均保持四次图捕获。BF16 和 FP8 的并发测量
区间分别重叠 56.31 和 47.45 秒，启动偏差为 0.37 和 3.06 ms。
双卡配对量化误差相同：MAE 0.004123、RMSE 0.033087、最大绝对误差 2.00924。

精简原始指标、逐卡配置、来源哈希与误差维度见
[evidence-20261010.json](evidence-20261010.json)。PaliGemma tokenizer 的 119 个
指令/状态/空白及 Unicode 探针与原 SentencePiece token IDs 完全一致，词表大小
257,152；使用隔离缓存与环境，没有修改共享模型环境。

历史 Triton 权重量化 B=1 配置为 121.76 ms、8.213 obs/s、5.89 GiB，预热十条且
关闭 Inductor。该结果只作历史记录，不用作新 FP8 的配对加速比基线。

在本 benchmark 根目录执行，先修改模型和数据路径：

```bash
python benchmark.py --config 4090/bf16/config.json
python benchmark.py --config 4090/fp8/config.json
python compare.py 4090/bf16/runs/libero10-b4.result.json \
  4090/fp8/runs/libero10-b4.result.json --output 4090/fp8/runs/comparison.json
```

数据、逐条动作、运行日志及训练环境保留在实验机
`/mnt/zhouzhenyuan/embodiinfer-optimization-20261010/pi05/`。
完整结果只保存在实验产物中，仓库保留精简指标与可复现配置。
