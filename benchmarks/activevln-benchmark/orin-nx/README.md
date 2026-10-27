# ActiveVLN · Orin NX

R2R val-unseen，BF16 / 校准 INT8 各 256 集、batch=1；Orin NX 推理，RTX 4090 渲染。
两组使用相同的按场景比例抽样清单，配置见 [config.json](config.json)。

```bash
BENCH=benchmarks/activevln-benchmark/orin-nx
DATASET=r2r
python "$BENCH/run.py" bf16 --dataset "$DATASET" \
  --checkpoint "$CKPT" --data-root "$WARMUP_R2R" \
  --manifest "$MANIFESTS/selected-$DATASET-navigation-256.json" --output "$RUN_ROOT"
# INT8：把 bf16 改为 int8，使用新的输出目录，并加 --calibration "$CALIBRATION"
```

`CKPT` 使用已准备的 BF16 快照；`CALIBRATION` 是保留的 `calibration-maxima.pt`，
脚本校验哈希。INT8 量化 204 个文本/视觉 MLP 层，其余保持 BF16。

转发 NX 服务端口，将该组输出的配置、ready、token 和 episode 清单复制到模拟器机器的
`RUN_ROOT`，然后运行：

```bash
EMBODIRUN=/path/to/EmbodiRun HABITAT_PYTHON=/path/to/habitat/python \
SCENES=/path/to/scene_datasets SIMULATOR_GPU=0 \
  bash "$BENCH/navigate.sh" "$DATASET" "$RUN_ROOT" http://127.0.0.1:18767 "$RUN_ROOT/navigation"
```

| 精度 | 完成集数 | 模型 E2E mean / P95 | SR | SPL | NE / Oracle SR | 峰值 CUDA allocated / reserved | 整集耗时合计 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BF16 | 256 / 256 | 672.31 / 956.86 ms | 46.09% | 0.4184 | 5.838 m / 50.00% | 9.67 / 10.31 GiB | 90.71 min |
| INT8 | 256 / 256 | 505.42 / 786.50 ms | 46.48% | 0.4148 | 5.720 m / 51.56% | 6.96 / 8.01 GiB | 65.05 min |

INT8 平均模型 E2E 降低 **24.8%**，SR 增加 0.39 个百分点，SPL 降低 0.0036。
两组均无 OOM、非法动作或图执行回退。

模型 E2E 从已解码 CPU RGB 到 CPU actions，不含预热、HTTP 和模拟器；整集耗时为
256 集实际耗时之和。两组闭环轨迹、调用数不同（BF16 6480 / INT8 5858），耗时差异不是固定输入的加速比。
完整 HTTP 耗时、逐集 token/action 轨迹与日志保留在本地，不提交 Git。
