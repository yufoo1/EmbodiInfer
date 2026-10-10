# 0008 — Measured FP8 inference on Ada

- Status: Implemented; lossy offline validation
- Author: yufoo1
- Date: 2026-10-10

## 1. Summary

Implement graph-safe dynamic FP8 activation conversion in the backend and measure
native W8A8 against optimized BF16 for complete PI0.5 inference. Preserve existing
quantization configuration and policy-local precision exclusions.

## 2. Motivation and current gap

The published 4090 PI0.5 profile uses weight-only Triton FP8. It reduces memory but
takes 121.76 ms per observation at B=1. `backend/torch/fp8.py` additionally provides
native W8A8, but its dynamic activation conversion launches several Torch kernels.
Its pre-SM100 tensorwise branch incorrectly pairs rowwise activation scales with a
scalar weight scale. Current Torch 2.13 on SM89 accepts both consistently rowwise
and consistently tensorwise `_scaled_mm` inputs.

## 3. Goals and non-goals

Reduce measured memory and complete CPU-observation-to-CPU-action time. Target
25 observations/s per GPU after actual batch occupancy is accounted for, retaining
all ten denoising steps and complete 50-by-7 action chunks. This is not a claim
about closed-loop task accuracy, HTTP latency or aggregate multi-GPU capacity.

## 4. Design

Keep E4M3 weights and FP32 scales. Fuse rowwise activation maximum, scaling and
conversion into one optional Triton kernel; native GEMM continues through the
registered Torch backend. Retain the Torch conversion fallback when Triton cannot
run. Select tensorwise activation conversion when the weight has a scalar scale.
Use existing `ignored_layers` for measured prefix/expert precision combinations.
The tensorwise converter uses block maxima, one scalar reduction and conversion,
avoiding full-size FP32 temporaries. Both converters preserve Torch rounding;
Ada's FP16 conversion intermediary uses round-to-odd to avoid false E4M3 ties.
Native FP8 may retain Inductor for the unquantized vision encoder and arithmetic
helpers. A prefix containing quantized projections runs its registered kernels
inside the existing CUDA Graph instead of tracing Python quantization dispatch
into one Inductor graph. Unquantized prefixes keep their compiled encoder.

The alternative weight-only kernel remains available on older GPUs. It saves
weight storage but does not execute FP8 tensor-core matrix multiplication. Static
activation calibration is a possible later experiment, not part of dynamic FP8.

## 5. Model-agnosticism verdict

Activation conversion and scale routing are backend operations with no policy or
engine imports. PI0.5 measurement and selective projection configuration belong
to its benchmark. No engine scheduling changes are required.

## 6. Losslessness and precision criterion

FP8 is explicitly authorized lossy quantization. Compare complete physical action
chunks against optimized BF16 at identical batch shape, sample identities,
per-sample noise seed, ten steps and `highest` matmul precision. Report MAE, RMSE,
maximum error and each action dimension; do not declare task-quality equivalence
from these errors. Compare the fused converter byte-for-byte with Torch dynamic
E4M3 conversion on finite BF16 inputs, including zero rows and rounding boundaries.
Use exact division where needed to preserve the original rounding contract.

## 7. Implementation plan

Add an optional activation conversion kernel under `backend/triton`, correct
native scale routing, and extend the existing PI0.5 quantization benchmark for
batch occupancy. Preserve existing weight-only and unquantized profiles.

## 8. Test plan

CPU tests cover configuration and scalar/row scale dispatch. GPU tests cover
conversion, noncontiguous input, zero rows, native GEMM and CUDA Graph replay with
changed inputs. Real-weight runs check finite full action chunks and record drift.

## 9. Benchmark plan

Two RTX 4090 24 GiB cards, independent processes, BF16 activations with selected
FP8 projections, B=1 and B=4 initially, fixed LIBERO-10 selection of 1,600 observations and
complete shape warmup. First compare profiles on one GPU without concurrent
training, then repeat the winning configuration on both. Record allocated and
reserved memory, preprocessing, complete model and E2E time, actual backend,
graph-capture counts, checkpoint and source identity. A successful profile must
beat the same-batch optimized BF16 baseline in E2E and memory, and reach 25/s.

## 10. Risks and limitations

Small GEMMs may spend more time converting activations than they save. Quantizing
the action expert may amplify output drift over ten steps. Graph capture retains
workspace and can offset weight-memory savings. CUDA/Torch versions can support
different scale layouts, so runtime validation and explicit errors remain needed.

## 11. ActiveVLN extension

Expose the same optional `quantization` configuration on ActiveVLN. Quantize only
the seven named text projections in each Qwen decoder layer; leave vision,
embeddings, norms and the language head in their checkpoint precision. Policy-local
names `text.layers.<index>.{self_attn,mlp}.<projection>` support exclusions. The
initial Ada candidate excludes attention projections and uses tensorwise native
W8A8 for the 108 MLP projections. Small K/V GEMMs and row-scaled large GEMMs were
slower in a Torch 2.10/cu128 shape probe. Backend registration and generic linear
storage stay unchanged.

FP8 is an explicitly lossy inference option. Reject the BF16-only serial-reference
draft verifier and FP32 static-tree projection when quantized layers are present.
Ordinary draft verification can execute the quantized target, but does not inherit
the BF16 serial-reference parity claim. Record every quantized layer and its
resolved backend in replay and HTTP startup evidence.

Measure complete R2R/RxR replay on independent 4090 instances with actual batch
occupancy, retaining vision, preprocessing, full histories and CPU action transfer.
Evaluate navigation quality separately in downstream EmbodiRun/Habitat through
the versioned HTTP API, using paired BF16/FP8 manifests, dataset-specific prompts
and actions, and explicit STOP/geodesic success rules. Report SR and SPL; token
agreement is not a substitute. Simulator execution and task metrics remain outside
EmbodiInfer.
