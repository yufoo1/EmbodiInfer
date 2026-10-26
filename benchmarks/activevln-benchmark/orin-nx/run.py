"""Start the measured BF16 or calibrated INT8 service in an existing Jetson environment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import sys
from pathlib import Path

DIRECTORY = Path(__file__).resolve().parent
BENCHMARK = DIRECTORY.parent
REPOSITORY = BENCHMARK.parents[1]


def main() -> None:
    """Resolve asset paths and launch one profile without changing system settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("precision", choices=("bf16", "int8"))
    parser.add_argument("--checkpoint", type=Path, required=True, help="Prepared BF16 checkpoint")
    parser.add_argument("--data-root", type=Path, required=True, help="R2R annotations and RGB frames")
    parser.add_argument("--output", type=Path, required=True, help="New directory for this service run")
    parser.add_argument("--calibration", type=Path, help="Retained calibration-maxima.pt for INT8")
    parser.add_argument("--port", type=int, default=18767)
    parser.add_argument("--prepare-only", action="store_true", help="Write configuration without GPU work")
    args = parser.parse_args()
    if (args.precision == "int8") != (args.calibration is not None):
        parser.error("--calibration is required for INT8 and must be omitted for BF16")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    config = json.loads((DIRECTORY / "config.json").read_text())
    profile = config.pop("int8_profile")
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    data = args.data_root.expanduser().resolve(strict=True)
    if not (checkpoint / "config.json").is_file():
        raise FileNotFoundError("Checkpoint must contain config.json")
    annotations = data / "annotations_v1-3.json"
    records = json.loads(annotations.read_text())
    selected = sorted(records, key=lambda row: (int(row["id"]), row["video"]))[:48]
    identities = [{"episode_id": int(row["id"]), "frames": len(row["actions"])} for row in selected]
    digest = hashlib.sha256(json.dumps(identities, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    if digest != config["datasets"][0]["selection_sha256"]:
        raise ValueError("R2R selection differs from the measured 48 episodes")
    calibration = None
    if args.calibration is not None:
        calibration = args.calibration.expanduser().resolve(strict=True)
        if hashlib.sha256(calibration.read_bytes()).hexdigest() != profile["calibration_sha256"]:
            raise ValueError("Calibration differs from the measured INT8 profile")
        config["precision_experiment"] = profile
    output = args.output.expanduser().resolve()
    config.update(checkpoint=str(checkpoint), output_dir=str(output))
    config["datasets"][0].update(root=str(data), annotations=str(annotations))
    output.mkdir(parents=True, exist_ok=False)
    config_path = output / "config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    token = secrets.token_urlsafe(32)
    token_path = output / "service-token"
    with open(token_path, "x", opener=lambda path, flags: os.open(path, flags, 0o600)) as stream:
        stream.write(token + "\n")
    if args.prepare_only:
        print(f"Prepared {config_path}; model weights and RGB frames are checked when serving")
        return
    environment = dict(os.environ)
    environment.pop("PYTORCH_ALLOC_CONF", None)
    environment.update(
        PYTHONPATH=os.pathsep.join(
            filter(None, (str(REPOSITORY), str(BENCHMARK), environment.get("PYTHONPATH")))
        ),
        PYTHONDONTWRITEBYTECODE="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        OMP_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        PYTORCH_CUDA_ALLOC_CONF="backend:native,garbage_collection_threshold:0.7,max_split_size_mb:128",
        ACTIVEVLN_EVICT_CHECKPOINT="1",
        ACTIVEVLN_BENCHMARK_TOKEN=token,
    )
    cuda_include = Path(environment.get("CUDA_HOME", "/usr/local/cuda")) / "targets/aarch64-linux/include"
    if cuda_include.is_dir():
        environment["CPATH"] = os.pathsep.join(filter(None, (str(cuda_include), environment.get("CPATH"))))
    command = [
        sys.executable,
        "-u",
        str(DIRECTORY / "serve.py"),
        "--config",
        str(config_path),
        "--source",
        str(REPOSITORY),
        "--ready-file",
        str(output / "service-ready.json"),
        "--streamed-loading",
        "--port",
        str(args.port),
        "--max-wait-ms",
        "0",
    ]
    if calibration is not None:
        command.extend(["--calibration", str(calibration)])
    os.execve(sys.executable, command, environment)


if __name__ == "__main__":
    main()
