"""Collect disjoint teacher trajectories and train a feature-conditioned token draft."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from benchmark import load_navigation, make_observation, provenance, read_rgb
from benchmark_batch import ReplaySlot
from torch.nn import functional as F

from embodiinfer.policies import make_policy
from embodiinfer.policies.activevln.draft_activevln import ActiveVLNDraft


def save_json(path: Path, value: dict) -> None:
    """Atomically publish progress and provenance independently of large tensor shards."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def collect(config: dict, output: Path) -> None:
    """Record the target features immediately preceding every actually selected token."""
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(config["cpu_threads"])
    torch.manual_seed(config["seed"])
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    evaluation = load_navigation(config["evaluation_dataset"])
    candidates = load_navigation(config["training_dataset"])
    held_out_images = {hashlib.sha256(p.read_bytes()).digest() for ep in evaluation for p in ep.frames}
    excluded = []
    episodes = []
    for episode in candidates:
        if any(hashlib.sha256(p.read_bytes()).digest() in held_out_images for p in episode.frames):
            excluded.append(episode.episode_id)
        else:
            episodes.append(episode)
    if len(episodes) < 10:
        raise ValueError("too few training episodes after evaluation-image exclusion")
    validation_ids = {ep.episode_id for ep in episodes[-config["validation_episodes"] :]}
    manifest = {
        "schema": "activevln.draft_teacher.v1",
        "status": "loading",
        "config": config,
        "environment": provenance(device),
        "held_out_dataset": config["evaluation_dataset"],
        "held_out_episode_ids": [ep.episode_id for ep in evaluation],
        "excluded_image_overlap_episodes": excluded,
        "validation_episode_ids": sorted(validation_ids),
        "training_episode_ids": [ep.episode_id for ep in episodes if ep.episode_id not in validation_ids],
        "selection": [
            {"episode_id": ep.episode_id, "frames": len(ep.frames), "video": ep.video} for ep in episodes
        ],
        "shards": [],
        "observations": 0,
        "tokens": 0,
    }
    save_json(output / "manifest.json", manifest)
    policy = (
        make_policy(
            "activevln",
            checkpoint=config["checkpoint"],
            revision=config["checkpoint_revision"],
            attention="sdpa",
            max_new_tokens=512,
            max_context=128000,
            do_sample=False,
            repetition_penalty=1.05,
            action_space="r2r",
        )
        .to(device=device, dtype=torch.bfloat16)
        .eval()
    )
    size = config["batch_size"]
    captured = []
    with torch.inference_mode():
        runtime = policy.create_batched_runtime(
            batch_size=size,
            workspace_tokens=65536,
            query_bucket_size=1,
            cuda_graph=True,
            fused_ops=True,
            split_attention=True,
            tree_decode=False,
            kv_pool_tokens=147456,
        )
        runtime.prewarm(
            [make_observation(read_rgb(ep.frames[0]), ep.instruction) for ep in episodes],
            context_buckets=(512, 1024, 2048, 4096, 8192, 16384, 32768, 65536),
        )
        hook = policy._lm_head.register_forward_pre_hook(
            lambda module, args: captured.append(args[0].clone())
        )
        manifest["status"] = "collecting"
        pending, slots, shard_rows = iter(episodes), [], []
        started = time.perf_counter()

        def flush() -> None:
            if not shard_rows:
                return
            filename = f"teacher-{len(manifest['shards']):04d}.pt"
            torch.save(shard_rows, output / filename)
            manifest["shards"].append(filename)
            manifest["elapsed_seconds"] = time.perf_counter() - started
            save_json(output / "manifest.json", manifest)
            print(
                json.dumps({k: manifest[k] for k in ("observations", "tokens", "elapsed_seconds")}),
                flush=True,
            )
            shard_rows.clear()

        try:
            while True:
                while len(slots) < size:
                    episode = next(pending, None)
                    if episode is None:
                        break
                    slots.append(ReplaySlot(episode))
                if not slots:
                    break
                observations = [
                    make_observation(read_rgb(s.episode.frames[s.step]), s.episode.instruction) for s in slots
                ]
                captured.clear()
                prepared = runtime.prepare(observations, [s.memory for s in slots])
                generations = runtime.generate(runtime.prefill(prepared))
                features = torch.stack(captured).cpu()
                for row, (slot, generation) in enumerate(zip(slots, generations, strict=True)):
                    tokens = generation.token_ids[0].cpu()
                    length = len(tokens)
                    if features.shape[0] < length:
                        raise RuntimeError("missing teacher features for an accepted token")
                    shard_rows.append(
                        {
                            "episode_id": slot.episode.episode_id,
                            "step": slot.step,
                            "validation": slot.episode.episode_id in validation_ids,
                            "hidden": features[:length, row].clone(),
                            "tokens": tokens,
                            "stop_reason": generation.stop_reason,
                        }
                    )
                    manifest["observations"] += 1
                    manifest["tokens"] += length
                    slot.memory = generation.memory
                    slot.step += 1
                slots = [s for s in slots if s.step < len(s.episode.frames)]
                del prepared, generations, features, observations, generation, slot
                captured.clear()
                if len(shard_rows) >= 64:
                    flush()
            flush()
        finally:
            hook.remove()
        manifest.update(status="complete", runtime=runtime.stats())
        save_json(output / "manifest.json", manifest)


def train(source: Path, output: Path, *, epochs: int, width: int, block_size: int, seed: int) -> None:
    """Fit multi-token heads; select the checkpoint only by disjoint validation loss."""
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("teacher collection must complete before training")
    rows = [r for name in manifest["shards"] for r in torch.load(source / name, weights_only=True)]
    vocabulary = sorted({int(t) for r in rows if not r["validation"] for t in r["tokens"]})
    model = ActiveVLNDraft(rows[0]["hidden"].shape[1], vocabulary, width=width, block_size=block_size).cuda()
    splits = {}
    for validation in (False, True):
        hidden, roots, labels = [], [], []
        for row in rows:
            if row["validation"] != validation or len(row["tokens"]) < 2:
                continue
            ids = row["tokens"]
            hidden.append(row["hidden"][:-1])
            roots.append(ids[:-1])
            targets = torch.full((len(ids) - 1, block_size - 1), -100, dtype=torch.long)
            for offset in range(1, min(block_size, len(ids))):
                targets[: len(ids) - offset, offset - 1] = ids[offset:]
            labels.append(targets)
        if not hidden:
            raise ValueError("training and validation splits must both have examples")
        targets = torch.cat(labels).cuda()
        valid = (targets >= 0) & (targets < model.lookup.numel())
        mapped = model.lookup[targets.clamp(0, model.lookup.numel() - 1)]
        valid &= mapped < len(vocabulary)
        mapped = torch.where(valid, mapped, -100)
        splits[validation] = (torch.cat(hidden).cuda().float(), torch.cat(roots).cuda(), mapped)
    del rows
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    log, best = [], float("inf")
    metadata = {
        "teacher_manifest_sha256": hashlib.sha256((source / "manifest.json").read_bytes()).hexdigest(),
        "teacher_manifest": manifest,
        "seed": seed,
        "epochs_requested": epochs,
        "parameters": sum(p.numel() for p in model.parameters()),
        "training_examples": len(splits[False][0]),
        "validation_examples": len(splits[True][0]),
    }
    for epoch in range(epochs):
        model.train()
        hidden, roots, targets = splits[False]
        order = torch.randperm(len(hidden), device="cuda")
        for indices in order.split(256):
            logits = model(hidden[indices], roots[indices])
            loss = F.cross_entropy(logits.flatten(0, 1), targets[indices].flatten())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        total_loss, count = 0.0, 0
        correct = torch.zeros(block_size - 1, device="cuda")
        totals = torch.zeros_like(correct)
        with torch.inference_mode():
            hidden, roots, targets = splits[True]
            for begin in range(0, len(hidden), 512):
                target = targets[begin : begin + 512]
                logits = model(hidden[begin : begin + 512], roots[begin : begin + 512])
                valid = target != -100
                total_loss += F.cross_entropy(logits.flatten(0, 1), target.flatten(), reduction="sum").item()
                count += int(valid.sum())
                correct += ((logits.argmax(-1) == target) & valid).sum(0)
                totals += valid.sum(0)
        result = {
            "epoch": epoch + 1,
            "validation_loss": total_loss / count,
            "validation_accuracy_by_offset": (correct / totals.clamp_min(1)).tolist(),
        }
        log.append(result)
        print(json.dumps(result), flush=True)
        if result["validation_loss"] < best:
            best = result["validation_loss"]
            torch.save(
                model.checkpoint({**metadata, "selected_epoch": epoch + 1, "validation": result}),
                output / "draft.pt",
            )
        save_json(output / "training.json", {**metadata, "epochs": log, "best_validation_loss": best})


def main() -> None:
    """Keep collection and learning explicit, reproducible, and outside model execution."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    gather = commands.add_parser("collect")
    gather.add_argument("--config", type=Path, required=True)
    gather.add_argument("--output", type=Path, required=True)
    fit = commands.add_parser("train")
    fit.add_argument("--source", type=Path, required=True)
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--epochs", type=int, default=40)
    fit.add_argument("--width", type=int, default=512)
    fit.add_argument("--block-size", type=int, default=16)
    fit.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.command == "collect":
        collect(json.loads(args.config.read_text()), args.output)
    else:
        train(
            args.source,
            args.output,
            epochs=args.epochs,
            width=args.width,
            block_size=args.block_size,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
