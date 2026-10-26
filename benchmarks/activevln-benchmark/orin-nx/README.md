# ActiveVLN · Orin NX

## 运行

使用已准备好的 Jetson Python 环境、BF16 权重和 R2R 数据，配置见 [config.json](config.json)。
在 NX 上依次启动两组服务，每组评测结束后 Ctrl-C 退出：

```bash
BENCH=benchmarks/activevln-benchmark/orin-nx
python "$BENCH/run.py" bf16 --checkpoint "$CKPT" --data-root "$R2R" --output results/nx-bf16
python "$BENCH/run.py" int8 --checkpoint "$CKPT" --data-root "$R2R" --output results/nx-int8 \
  --calibration "$CALIBRATION"
```

`CALIBRATION` 使用此前的 `calibration-maxima.pt`，脚本校验其哈希。
在模拟器机器转发 NX 的 `18767` 端口，将该组输出的 `config.json`、`service-ready.json`、
`service-token` 和保留的 `selected-r2r-navigation-48.json` 放入 `$RUN_ROOT`，然后运行：

```bash
EMBODIRUN=/path/to/EmbodiRun HABITAT_PYTHON=/path/to/habitat/python SCENES=/path/to/scene_datasets \
  bash "$BENCH/navigate.sh" "$RUN_ROOT" http://127.0.0.1:18767 "$RUN_ROOT/navigation"
```

导航循环与 SR/SPL 由 EmbodiRun 的 `benchmarks/activevln-navigation` 提供。
权重、校准文件、数据和日志不提交 Git。

## R2R 48 集结果

Orin NX 16GB，batch=1，resident KV；同一组训练集轨迹，排除 33 次 warmup。
模型为 `Arvil/Qwen2.5-VL-3B_rl_r2r_4000`，版本及运行开关见配置。
INT8 为 204 个文本/视觉 MLP 层的校准 W8A8，其余保持 BF16。

| 指标 | BF16 | INT8 |
| --- | ---: | ---: |
| SR | 68.75%（33/48） | 58.33%（28/48） |
| SPL | 0.6437 | 0.5522 |
| NE / Oracle SR | 4.679 m / 75.00% | 5.722 m / 66.67% |
| 模型调用数 | 930 | 986 |
| 模型 E2E mean / P95 | 663.06 / 900.64 ms | 512.73 / 718.09 ms |
| 峰值 CUDA allocated / reserved | 8.54 / 9.19 GiB | 5.84 / 6.89 GiB |

**INT8 平均延迟降低 22.7%，SR 下降 10.42 个百分点，默认仍用 BF16。**
排除参与校准的第 1 集后，成功数为 32/47 对 27/47。两组均无 OOM、无执行 fallback；
各自与此前相同精度的 48 条 token/action 轨迹完全一致。
E2E 从已解码 CPU RGB 到 CPU actions，不含磁盘、warmup、HTTP 和模拟器；
闭环轨迹不同，这不是固定输入的加速比，也不代表 val-unseen 或 RxR 精度。
