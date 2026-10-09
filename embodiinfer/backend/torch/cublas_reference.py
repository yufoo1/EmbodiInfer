"""Optional cuBLASLt plans retaining a reference batch's split/reduction settings.

This is an inference backend, not a cross-version numerical guarantee. Callers
must validate their hardware, library version, shapes and chosen tuning profile.
An unsupported grouped projection uses actual serial Torch calls instead.
"""

from __future__ import annotations

import ctypes
import functools
import hashlib
import os
import shutil
import subprocess
import weakref
from pathlib import Path

import torch
from torch.nn import functional as F


def _cuda_paths() -> tuple[Path, Path, Path, str]:
    from importlib.metadata import PackageNotFoundError, distribution

    try:
        package = distribution("nvidia-cublas-cu12")
    except PackageNotFoundError as exc:
        raise RuntimeError("reference projections require the CUDA 12 cuBLAS developer package") from exc
    root = Path(package.locate_file("nvidia/cublas"))
    include, library = root / "include", root / "lib/libcublasLt.so.12"
    candidates = [Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")) / "include"]
    try:
        runtime = distribution("nvidia-cuda-runtime-cu12")
        candidates.append(Path(runtime.locate_file("nvidia/cuda_runtime/include")))
    except PackageNotFoundError:
        pass
    cuda = next((path for path in candidates if (path / "cuda_runtime_api.h").is_file()), None)
    if cuda is None or not (include / "cublasLt.h").is_file() or not library.is_file():
        raise RuntimeError(
            "reference projections require CUDA headers (CUDA_HOME) and cuBLASLt headers/library"
        )
    return include, cuda, library, package.version


@functools.lru_cache(maxsize=1)
def _load_library() -> ctypes.CDLL:
    import fcntl

    include, cuda, library, version = _cuda_paths()
    compiler = shutil.which(os.environ.get("CXX", "g++"))
    if compiler is None:
        raise RuntimeError("reference projections require a C++ compiler; set CXX or install g++")
    source = Path(__file__).with_name("_cublas_reference.cpp")
    identity = source.read_bytes() + Path(__file__).read_bytes()
    identity += f"{torch.__version__}:{version}:{compiler}:{include}:{cuda}:{library}".encode()
    digest = hashlib.sha256(identity).hexdigest()[:24]
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "embodiinfer/cublas-reference"
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / f"{digest}.so"
    with (cache / f"{digest}.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if not target.is_file():
            temporary = cache / f"{digest}.{os.getpid()}.so"
            command = [
                compiler,
                "-std=c++17",
                "-O2",
                "-shared",
                "-fPIC",
                f"-I{include}",
                f"-I{cuda}",
                str(source),
                str(library),
                f"-Wl,-rpath,{library.parent}",
                "-o",
                str(temporary),
            ]
            try:
                result = subprocess.run(command, capture_output=True, text=True, check=False)
                if result.returncode:
                    raise RuntimeError(f"cuBLASLt reference bridge build failed:\n{result.stderr[-8000:]}")
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
    loaded = ctypes.CDLL(str(target))
    loaded.reference_plan_create.argtypes = [ctypes.c_int] * 4 + [ctypes.c_void_p] + [ctypes.c_int] * 3
    loaded.reference_plan_create.restype = ctypes.c_void_p
    loaded.reference_plan_execute.argtypes = [ctypes.c_void_p] * 6
    loaded.reference_plan_execute.restype = ctypes.c_int
    loaded.reference_plan_destroy.argtypes = [ctypes.c_void_p]
    loaded.reference_plan_destroy.restype = None
    return loaded


def _release_plans(library: ctypes.CDLL, plans: dict) -> None:
    for pointer, _, _ in plans.values():
        library.reference_plan_destroy(pointer)
    plans.clear()


class ReferenceBatchLinear:
    """Project BQH blocks with the reduction settings selected for BH reference calls.

    Plans and workspace are private to the caller's CUDA runtime/stream. This
    object does not mutate modules or global backend settings. Keep it alive
    while captured graphs can replay its workspace. ``tuning`` selects a tested
    (algorithm, tile, stages) tuple; None retains cuBLASLt's heuristic selection.
    """

    def __init__(self, device: torch.device) -> None:
        if device.type != "cuda":
            raise ValueError("reference projections require a CUDA device")
        self._library = _load_library()
        self._workspace = torch.empty(33554432, device=device, dtype=torch.uint8)
        self._plans: dict[tuple, tuple[int, torch.Tensor, torch.Tensor | None]] = {}
        self._fallbacks: set[tuple] = set()
        self._finalizer = weakref.finalize(self, _release_plans, self._library, self._plans)

    @property
    def serial_fallback_shapes(self) -> int:
        """Number of unsupported plan shapes handled with actual serial projections."""
        return len(self._fallbacks)

    def project(
        self,
        inputs: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        *,
        tuning: tuple[int, int, int] | None = None,
    ) -> torch.Tensor:
        """Return BQN BF16 output; unsupported grouped plans preserve the BH calls."""
        tensors = (inputs, weight) if bias is None else (inputs, weight, bias)
        if (
            torch.is_grad_enabled()
            or inputs.ndim != 3
            or weight.ndim != 2
            or min(inputs.shape) < 1
            or min(weight.shape) < 1
            or inputs.shape[-1] != weight.shape[1]
            or (bias is not None and bias.shape != (weight.shape[0],))
            or any(
                tensor.device != self._workspace.device
                or tensor.dtype != torch.bfloat16
                or not tensor.is_contiguous()
                for tensor in tensors
            )
        ):
            raise ValueError(
                "reference projections require contiguous CUDA BF16 BQH inputs and matching weights"
            )
        batch, query, width = inputs.shape
        key = (
            weight.data_ptr(),
            0 if bias is None else bias.data_ptr(),
            tuple(weight.shape),
            batch,
            query,
            tuning,
        )

        def serial() -> torch.Tensor:
            return torch.stack(
                [F.linear(inputs[:, index].contiguous(), weight, bias) for index in range(query)], 1
            )

        if key in self._fallbacks:
            return serial()
        if key not in self._plans:
            selected = (-1, -1, -1) if tuning is None else tuning
            with torch.cuda.device(inputs.device):
                pointer = self._library.reference_plan_create(
                    width,
                    weight.shape[0],
                    batch,
                    batch * query,
                    None if bias is None else bias.data_ptr(),
                    *selected,
                )
            if not pointer:
                raise RuntimeError("cuBLASLt returned an empty reference plan")
            self._plans[key] = pointer, weight, bias
        output = inputs.new_empty(batch, query, weight.shape[0])
        with torch.cuda.device(inputs.device):
            status = self._library.reference_plan_execute(
                self._plans[key][0],
                inputs.data_ptr(),
                weight.data_ptr(),
                output.data_ptr(),
                self._workspace.data_ptr(),
                torch.cuda.current_stream(inputs.device).cuda_stream,
            )
        if status == 15:  # CUBLAS_STATUS_NOT_SUPPORTED
            self._fallbacks.add(key)
            return serial()
        if status:
            raise RuntimeError(f"cuBLASLt reference projection failed with status {status}")
        return output
