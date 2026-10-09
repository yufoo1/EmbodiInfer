# PI0.5 · 4090 · BF16

当前 [config.json](config.json) 为 B=4 配对基线：原生优化、Inductor、
prefix/denoise CUDA Graph、Triton attention、低内存加载，四个 CPU 线程。
400 次预热 batch 调用覆盖全部 1,600 条 LIBERO-10 观测；保留完整十步去噪。
详见 [方法](../../README.md) 和 [FP8 对照](../fp8/README.md)。

2026-10-10 单卡完整回放，吞吐按实际观测数摊销，显存为 CUDA allocated 峰值：

| Batch | 观测数 | E2E ms/观测 | Forward ms/观测 | obs/s/卡 | 显存 GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 1,600 | 42.83 | 38.41 | 23.35 | 10.80 |
| 4 | 1,600 | 29.74 | 25.81 | 33.63 | 11.16 |

双卡同步复测中，两卡各测 1,600 条观测，GPU 0/1 分别为 33.54/33.64 obs/s，
平均每卡 33.59 obs/s，E2E 29.77 ms/观测，显存仍为 11.16 GiB/卡。
对应 FP8 平均每卡 41.59 obs/s、6.91 GiB；详细条件与量化误差见 FP8 对照。

旧版 B=1 历史记录为 E2E 39.77 ms、Forward 36.74 ms、25.145 obs/s、10.80 GiB；
不能与本轮 FP8 直接组成配对比较。

从 benchmark 根目录运行，修改模型和数据路径后新输出进入本目录 `runs/`：

```bash
python benchmark.py --config 4090/bf16/config.json
```
