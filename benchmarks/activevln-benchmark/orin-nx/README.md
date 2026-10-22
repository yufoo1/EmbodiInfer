# ActiveVLN · Orin NX

环境版本、模型/数据 revision 和推理设置见 [config](configs/r2r-b1.json)。

## 运行

在匹配配置的 Jetson Python 环境中，从仓库根目录执行。
R2R 目录需包含 `annotations_v1-3.json` 和前 48 个 episode 的 `images/`；
可用 [prepare_data.py](../../streamvln-benchmark/prepare_data.py) 提取。

```bash
export NX_BENCH=benchmarks/activevln-benchmark/orin-nx

# 首次准备 BF16 权重；已有转换后的目录则跳过。
python3 "$NX_BENCH/scripts/prepare_checkpoint.py" convert \
  --source /path/to/r2r-fp32 --destination /path/to/r2r-bf16

python3 "$NX_BENCH/scripts/run.py" \
  --checkpoint /path/to/r2r-bf16 --data-root /path/to/R2R \
  --profile full32k --output-dir "$NX_BENCH/runs/r2r-full32k"

python3 "$NX_BENCH/scripts/summarize.py" "$NX_BENCH/runs/r2r-full32k"
```

先检查能否跑通可将 `--profile` 改为 `smoke`，每次使用新输出目录。
报告与日志保存在本地 `runs/`，不提交 Git。

## 结果

R2R 离线回放结果：

| 指标 | 结果 |
| --- | ---: |
| 完成 episodes / 帧 | **48 / 2,997** |
| E2E mean / P50 / P95 | **662.47 / 653.35 / 831.98 ms** |
| Model-forward mean | 647.74 ms |
| 吞吐（帧数 / 累计 E2E） | 1.5095 次/秒 |
| CUDA allocated / reserved 峰值 | 9.949 / 11.025 GiB |
| 整个进程系统 RAM / swap 峰值 | 13.736 GiB / 71 MiB |
| 内存分配失败 / 执行路径回退 | 0 / 0 |

E2E 从已解码 CPU RGB 到 CPU action，不含读盘、启动/预热及机器人执行。
系统内存为 1 秒采样；统一内存下 CUDA 与系统 RAM 峰值不能相加。
