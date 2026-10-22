# ActiveVLN · Orin NX · INT8

校准文本和视觉 block MLP 的 W8A8 量化，使用融合缩放；其余层保持 BF16。
模型、环境与基础配置见 [BF16](../bf16/README.md)。

## 运行

先按 BF16 说明准备权重和 R2R 数据，从仓库根目录执行：

```bash
export NX_BENCH=benchmarks/activevln-benchmark/orin-nx
export NX_INT8="$NX_BENCH/int8/runs/probe"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python3 "$NX_BENCH/int8/scripts/precision.py" \
  --checkpoint /path/to/r2r-bf16 --data-root /path/to/R2R --output-dir "$NX_INT8" \
  --variants smooth_both
python3 "$NX_BENCH/bf16/scripts/run_with_streamed_loading.py" \
  --config "$NX_INT8/replay-config.json" --output "$NX_INT8/bf16.json"
python3 "$NX_BENCH/int8/scripts/precision.py" replay \
  --config "$NX_INT8/replay-config.json" --calibration "$NX_INT8/calibration-maxima.pt" \
  --variant smooth_both --output "$NX_INT8/int8.json"
python3 "$NX_BENCH/int8/scripts/compare.py" "$NX_INT8/bf16.json" "$NX_INT8/int8.json" \
  --allow-probe --output "$NX_INT8/comparison.json"
```

校准使用第 1 集前 12 帧，短回放使用第 2–4 集各 8 帧，各版本独立累积历史。
[未校准文本 INT8 配置](configs/r2r-b1-uncalibrated.json) 仅用于直接量化实验；
这里的校准方案使用 BF16 配置加 `calibration-maxima.pt` 和 `smooth_both`。
每次使用新输出目录，`runs/` 不提交 Git。

## 短回放结果

| 方案 | E2E mean | 与 BF16 完整动作串一致 |
| --- | ---: | ---: |
| BF16 | 699.88 ms | 24/24 |
| 校准文本 + 视觉 INT8 | 497.76 ms | 20/24 |

E2E 从已解码 CPU RGB 到 CPU action，不含读盘、启动/预热及机器人执行。
动作一致率不是导航成功率。

## 48 集闭环导航

使用 [serve_batch.py](../../serve_batch.py) 与 EmbodiRun 的
`benchmarks/activevln-navigation/benchmark.py --endpoint ...`，模型端为 NX、batch=1。
两组使用相同 BF16 基础配置和 manifest；INT8 服务增加
`--calibration calibration-maxima.pt --variant smooth_both`。
运行命令及进程内分配器设置见 EmbodiRun `benchmarks/README.md` 的
“Separate simulator and NX inference hosts”。

| 指标 | BF16 基线 | 校准文本 + 视觉 INT8 |
| --- | ---: | ---: |
| 成功集数 / 48 | 33 | 28 |
| SR | 68.75% | 58.33% |
| SPL | 0.6437 | 0.5522 |
| 导航误差 NE | 4.679 m | 5.722 m |
| Oracle SR | 75.00% | 66.67% |
| 模型调用次数 | 930 | 986 |
| 模型端 E2E mean / P95 | 669.96 / 914.56 ms | 514.84 / 726.24 ms |

两组均完成 48 集，无 OOM、无非法动作；初始观测/位姿一致，96 条执行轨迹校验通过。
排除校准用的第 1 集后，成功集数仍从 32/47 降至 27/47。
这些是选定训练轨迹的回归结果；闭环路径和调用次数不同，耗时不能当作固定输入加速比。
**INT8 的 SR 下降 10.42 个百分点，暂保留为实验选项，默认仍用 BF16。**
