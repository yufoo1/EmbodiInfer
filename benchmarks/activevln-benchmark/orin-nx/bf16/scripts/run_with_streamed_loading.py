"""Run the existing batch benchmark with the measured Jetson loading adaptation."""

from __future__ import annotations

import gc
import os
import runpy
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def streamed_loading() -> Iterator[None]:
    """Temporarily stream BF16 shards to CUDA without changing execution settings."""
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration

    original = Qwen2_5_VLForConditionalGeneration.from_pretrained.__func__

    def evict_checkpoint_pages(checkpoint: str) -> None:
        if os.environ.get("ACTIVEVLN_EVICT_CHECKPOINT", "1") == "1":
            for path in Path(checkpoint).glob("model-*.safetensors"):
                with path.open("rb") as stream:
                    os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)

    @classmethod
    def load_to_cuda(cls, *args, **kwargs):
        kwargs.update(low_cpu_mem_usage=True, device_map={"": "cuda:0"}, torch_dtype=torch.bfloat16)
        checkpoint = args[0] if args else kwargs["pretrained_model_name_or_path"]
        evict_checkpoint_pages(checkpoint)
        model = original(cls, *args, **kwargs)
        gc.collect()
        evict_checkpoint_pages(checkpoint)
        torch.cuda.synchronize()
        if not all(
            p.device.type == "cuda" and (not p.is_floating_point() or p.dtype == torch.bfloat16)
            for p in model.parameters()
        ):
            raise RuntimeError("Expected all floating parameters to be BF16 on CUDA")
        print("STARTUP_ADAPTER: low_cpu_mem_usage=True, device_map=cuda:0, dtype=bfloat16", flush=True)
        return model

    Qwen2_5_VLForConditionalGeneration.from_pretrained = load_to_cuda
    try:
        yield
    finally:
        Qwen2_5_VLForConditionalGeneration.from_pretrained = classmethod(original)


def main() -> None:
    """Run the replay benchmark with the measured loading adaptation."""
    benchmark = Path(__file__).resolve().parents[3] / "benchmark_batch.py"
    sys.path.insert(0, str(benchmark.parent))
    with streamed_loading():
        runpy.run_path(str(benchmark), run_name="__main__")


if __name__ == "__main__":
    main()
