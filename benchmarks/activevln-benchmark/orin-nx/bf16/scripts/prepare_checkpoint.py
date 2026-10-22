"""Create or verify the pinned R2R BF16 snapshot without allocating CUDA memory."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

CONFIG = Path(__file__).resolve().parents[1] / "configs/r2r-b1.json"


def digest(path: Path) -> str:
    """Return a file SHA-256 while keeping the read buffer bounded."""
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()


def verify(checkpoint: Path) -> None:
    """Check every derived shard and auxiliary file against its conversion manifest."""
    manifest = json.loads((checkpoint / "conversion-manifest.json").read_text())
    expected = json.loads(CONFIG.read_text())
    if (manifest["source_repository"], manifest["source_revision"]) != (
        expected["checkpoint_repository"],
        expected["checkpoint_revision"],
    ):
        raise ValueError("Unexpected checkpoint repository/revision")
    source_hashes = expected["checkpoint_source_sha256"]
    if {row["name"]: row["source_sha256"] for row in manifest["shards"]} != source_hashes:
        raise ValueError("Source shard hashes differ from the pinned snapshot")
    for row in manifest["shards"] + manifest["files"]:
        path = checkpoint / row["name"]
        if path.parent != checkpoint or not path.is_file():
            raise ValueError(f"Missing or unsafe checkpoint file: {path}")
        size = row.get("derived_size", row.get("size"))
        sha = row.get("derived_sha256", row.get("sha256"))
        if path.stat().st_size != size or digest(path) != sha:
            raise ValueError(f"Checkpoint checksum mismatch: {path.name}")
    print("Verified pinned BF16 checkpoint", flush=True)


def convert(source: Path, destination: Path) -> None:
    """Verify pinned FP32 shards, cast on CPU, and check each saved tensor exactly."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    torch.set_num_threads(4)
    expected = json.loads(CONFIG.read_text())
    destination.mkdir(parents=True, exist_ok=False)
    for path in source.iterdir():
        if (
            path.is_file()
            and path.suffix in (".json", ".txt", ".model", ".jinja")
            and path.name != "conversion-manifest.json"
        ):
            shutil.copy2(path, destination / path.name)
    manifest = {
        "source_repository": expected["checkpoint_repository"],
        "source_revision": expected["checkpoint_revision"],
        "conversion": "Floating tensors cast to torch.bfloat16 on CPU; integer tensors unchanged.",
    }
    manifest.update(torch_version=torch.__version__, shards=[])
    total = 0
    for name, source_sha256 in expected["checkpoint_source_sha256"].items():
        path = source / name
        if digest(path) != source_sha256:
            raise ValueError(f"Source checksum mismatch: {path.name}")
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            tensors = {}
            for key in handle.keys():  # noqa: SIM118 -- safetensors.safe_open is not iterable.
                value = handle.get_tensor(key)
                tensors[key] = value.to(torch.bfloat16) if value.is_floating_point() else value.clone()
            metadata = handle.metadata()
        out = destination / path.name
        save_file(tensors, str(out), metadata=metadata)
        with safe_open(str(out), framework="pt", device="cpu") as handle:
            for key, value in tensors.items():
                if not torch.equal(value, handle.get_tensor(key)):
                    raise ValueError(f"Serialized tensor differs: {key}")
        total += sum(value.numel() * value.element_size() for value in tensors.values())
        manifest["shards"].append(
            {
                "name": path.name,
                "source_sha256": source_sha256,
                "derived_sha256": digest(out),
                "derived_size": out.stat().st_size,
                "tensors": len(tensors),
            }
        )
        del tensors, value
        print(f"Converted and verified {path.name}", flush=True)
    index = json.loads((source / "model.safetensors.index.json").read_text())
    index["metadata"]["total_size"] = total
    (destination / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")
    config = json.loads((source / "config.json").read_text())
    config["torch_dtype"] = "bfloat16"
    if isinstance(config.get("vision_config"), dict):
        config["vision_config"]["torch_dtype"] = "bfloat16"
    (destination / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    manifest["total_tensor_bytes"] = total
    manifest["files"] = [
        {"name": path.name, "size": path.stat().st_size, "sha256": digest(path)}
        for path in sorted(destination.iterdir())
        if path.suffix != ".safetensors"
    ]
    (destination / "conversion-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    verify(destination)


def main() -> None:
    """Convert a local source snapshot or verify an already transferred BF16 copy."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    conversion = commands.add_parser("convert")
    conversion.add_argument("--source", type=Path, required=True)
    conversion.add_argument("--destination", type=Path, required=True)
    verification = commands.add_parser("verify")
    verification.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    if args.command == "convert":
        convert(args.source.resolve(strict=True), args.destination.resolve())
    else:
        verify(args.checkpoint.resolve(strict=True))


if __name__ == "__main__":
    main()
