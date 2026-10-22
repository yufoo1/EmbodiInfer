"""Run one R2R batch-one profile with isolated caches and whole-process telemetry."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import shutil
import subprocess
import sys
import time
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

from prepare_checkpoint import digest, verify

DIRECTORY = Path(__file__).resolve().parents[1]
BENCHMARK = DIRECTORY.parents[1]
REPOSITORY = BENCHMARK.parents[1]


def validate_selection(data: Path) -> None:
    """Require the published 48-episode selection and all 2,997 consecutive JPEGs."""
    records = json.loads((data / "annotations_v1-3.json").read_text())
    ids = [int(row["id"]) for row in records]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate episode IDs in annotations")
    spec = json.loads((DIRECTORY / "configs/r2r-b1.json").read_text())["datasets"][0]
    selected = sorted(records, key=lambda row: (int(row["id"]), row["video"]))[: spec["episode_limit"]]
    actual = [{"episode_id": int(row["id"]), "frames": len(row["actions"])} for row in selected]
    selection_sha256 = hashlib.sha256(
        json.dumps(actual, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    if selection_sha256 != spec["selection_sha256"]:
        raise ValueError("Dataset selection differs from the measured 48 episodes / 2,997 frames")
    for row in selected:
        video = Path(row["video"])
        rgb = (data / video / "rgb").resolve(strict=True)
        if video.is_absolute() or ".." in video.parts or not rgb.is_relative_to(data):
            raise ValueError(f"Unsafe dataset path: {video}")
        frames = sorted(int(path.stem) for path in rgb.glob("*.jpg"))
        if frames != list(range(1, len(row["actions"]) + 1)):
            raise ValueError(f"Incomplete frames for episode {row['id']}")


def capture_hardware() -> dict[str, str]:
    """Read Jetson settings without changing clocks, power, swap, or global caches."""
    result = {}
    for name, path in {
        "model": "/proc/device-tree/model",
        "l4t": "/etc/nv_tegra_release",
    }.items():
        if Path(path).exists():
            result[name] = Path(path).read_text().strip("\x00\n")
    for directory in (
        Path("/sys/devices/17000000.ga10b/devfreq/17000000.ga10b"),
        Path("/sys/devices/platform/17000000.gpu/devfreq_dev"),
    ):
        if (directory / "max_freq").exists():
            result["gpu_max_frequency_hz"] = (directory / "max_freq").read_text().strip()
            result["available_gpu_frequencies_hz"] = (directory / "available_frequencies").read_text().strip()
            break
    for name, command in {
        "power_mode": ["nvpmodel", "-q"],
        "memory": ["free", "-b"],
        "swap": ["swapon", "--show", "--bytes"],
    }.items():
        if shutil.which(command[0]):
            process = subprocess.run(command, capture_output=True, text=True, check=False)
            result[name] = process.stdout.strip()
    return result


def main() -> int:
    """Resolve portable paths, verify inputs, and execute the existing benchmark once."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DIRECTORY / "configs/r2r-b1.json")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--data-root", type=Path, required=True, help="R2R root containing annotations and images/"
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="New directory; never overwrites a run"
    )
    parser.add_argument("--profile", choices=("full32k", "smoke", "full64k"), default="full32k")
    parser.add_argument(
        "--loader", choices=("streamed", "original", "streamed-cache-hot"), default="streamed"
    )
    parser.add_argument(
        "--prepare-only", action="store_true", help="Verify inputs and write config without GPU work"
    )
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve(strict=True)
    data = args.data_root.resolve(strict=True)
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Use a new output directory: {output}")
    verify(checkpoint)
    validate_selection(data)
    config = json.loads(args.config.read_text())
    config.update(checkpoint=str(checkpoint), output_dir=str(output / "reports"))
    config["datasets"][0].update(root=str(data), annotations=str(data / "annotations_v1-3.json"))
    if args.profile == "smoke":
        config.update(warmup_calls=3, max_steps_per_episode=3)
        config["datasets"][0]["episode_limit"] = 2
    capacity = {"full32k": 32768, "full64k": 65536, "smoke": 8192}[args.profile]
    config.update(graph_workspace_tokens=capacity, kv_pool_tokens=capacity)
    config["prewarm_context_buckets"] = [2**power for power in range(9, capacity.bit_length())]
    adapter = DIRECTORY / "scripts/run_with_streamed_loading.py"
    config["startup_adapter"] = {
        "mode": args.loader,
        "file": str(adapter) if args.loader != "original" else None,
        "sha256": digest(adapter) if args.loader != "original" else None,
        "checkpoint_page_eviction": args.loader == "streamed",
        "scope": "Adapter changes weight loading only; runtime and quantization follow the benchmark config.",
    }
    output.mkdir(parents=True)
    for name in ("reports", "logs", "cache/triton", "cache/cuda", "cache/huggingface", "tmp"):
        (output / name).mkdir(parents=True, exist_ok=True)
    config_path = output / "config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    (output / "logs/hardware-settings.json").write_text(json.dumps(capture_hardware(), indent=2) + "\n")
    if args.prepare_only:
        print(f"Inputs verified; config written to {config_path}")
        return 0
    environment = os.environ.copy()
    environment.update(
        PYTHONPATH=os.pathsep.join(
            filter(None, (str(REPOSITORY), str(BENCHMARK), environment.get("PYTHONPATH")))
        ),
        PYTHONDONTWRITEBYTECODE="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="4",
        TRITON_CACHE_DIR=str(output / "cache/triton"),
        CUDA_CACHE_PATH=str(output / "cache/cuda"),
        HF_HOME=str(output / "cache/huggingface"),
        TMPDIR=str(output / "tmp"),
        ACTIVEVLN_EVICT_CHECKPOINT="1" if args.loader == "streamed" else "0",
    )
    cuda = Path(environment.get("CUDA_HOME", "/usr/local/cuda-12.6"))
    environment["CUDA_HOME"] = str(cuda)
    for name, paths in {
        "CPATH": [str(cuda / "targets/aarch64-linux/include")],
        "LIBRARY_PATH": [str(cuda / "targets/aarch64-linux/lib"), "/usr/lib/aarch64-linux-gnu"],
    }.items():
        environment[name] = os.pathsep.join(filter(None, paths + [environment.get(name)]))
    entrypoint = BENCHMARK / "benchmark_batch.py" if args.loader == "original" else adapter
    run_name = f"r2r-b1-{args.profile}"
    command = [
        sys.executable,
        "-u",
        str(entrypoint),
        "--config",
        str(config_path),
        "--output",
        str(output / "reports" / f"{run_name}.json"),
    ]
    print(f"Running {args.profile}; output: {output}", flush=True)
    monitors = []
    with ExitStack() as stack:
        for tool, options, suffix in (
            ("tegrastats", ["--interval", "1000"], "tegrastats"),
            ("vmstat", ["-w", "1"], "vmstat"),
        ):
            if shutil.which(tool):
                stream = stack.enter_context((output / "logs" / f"{run_name}.{suffix}").open("w"))
                monitors.append(subprocess.Popen([tool, *options], stdout=stream, stderr=subprocess.STDOUT))
            else:
                print(f"Telemetry unavailable: {tool}", file=sys.stderr)
        log = stack.enter_context((output / "logs" / f"{run_name}.log").open("w"))
        started = time.monotonic()
        (output / "logs" / f"{run_name}.start").write_text(datetime.now(timezone.utc).isoformat() + "\n")
        try:
            process = subprocess.run(
                command, cwd=REPOSITORY, env=environment, stdout=log, stderr=subprocess.STDOUT, check=False
            )
            usage = resource.getrusage(resource.RUSAGE_CHILDREN)
            resources = {
                "returncode": process.returncode,
                "elapsed_seconds": time.monotonic() - started,
                "max_rss_kib": usage.ru_maxrss,
                "user_seconds": usage.ru_utime,
                "system_seconds": usage.ru_stime,
                "major_page_faults": usage.ru_majflt,
                "minor_page_faults": usage.ru_minflt,
            }
            (output / "logs" / f"{run_name}.resource.json").write_text(json.dumps(resources, indent=2) + "\n")
            (output / "logs" / f"{run_name}.exit").write_text(str(process.returncode) + "\n")
        finally:
            for monitor in monitors:
                if monitor.poll() is None:
                    monitor.terminate()
                monitor.wait()
            (output / "logs" / f"{run_name}.end").write_text(datetime.now(timezone.utc).isoformat() + "\n")
    return process.returncode if process.returncode >= 0 else 128 - process.returncode


if __name__ == "__main__":
    raise SystemExit(main())
