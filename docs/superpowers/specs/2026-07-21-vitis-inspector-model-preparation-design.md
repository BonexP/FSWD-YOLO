# Vitis Inspector Model Preparation Design

## Goal

Reduce avoidable CPU partitions reported by Vitis AI 3.5 for FSWD-YOLO without changing the checkpoint, training behavior, learned parameters, or installed Vitis AI environment.

## Evidence

The first successful inspection for `DPUCZDX8G_ISA1_B4096` produced six device subgraphs, including two DPU subgraphs. The report identifies two primary causes of CPU assignment:

- `aten::silu_` cannot be converted to XIR because Ultralytics enables in-place activation during model initialization.
- `nndct_strided_slice` cannot be assigned to this DPU because `C2f`-derived modules use `chunk(2, 1)` in their normal forward path.

CPU constants, permutes, and concats are mostly downstream effects of those unsupported nodes.

## Approaches

1. **Prepare only the Inspector model in memory (selected).** Disable `nn.SiLU.inplace` and bind `C2f.forward_split` before NNDCT tracing. This matches Ultralytics' own non-TensorFlow export preparation while leaving training and checkpoints unchanged.
2. **Change global model definitions.** This would also affect training, validation, and existing checkpoints' execution behavior, creating unnecessary regression risk.
3. **Rewrite the NNDCT graph after tracing.** This depends on private compiler internals and would hide incompatibilities rather than presenting a normal PyTorch graph to the supported API.

## Design

`scripts/inspect_fswd_vitis.py` will expose a dependency-injected preparation helper. It walks `model.modules()` and:

- sets each in-place `nn.SiLU` to `inplace=False`
- binds each `C2f` instance with a callable `forward_split` to that method
- returns stable counts for both changes

Before preparation, the CLI evaluates the original model on the same deterministic dummy tensor used by Inspector. It evaluates the prepared model again, requires equal output shapes and `torch.allclose` with `rtol=1e-5`, `atol=1e-6`, and records maximum and mean absolute error. A mismatch stops inspection and is written to the failure manifest.

The manifest records the preparation counts and parity metrics. This allows the next Vitis run to distinguish a graph-compatibility transformation from a model-semantic change.

## Verification

Dependency-free unit tests verify that only in-place SiLU modules change, `C2f` modules bind `forward_split`, deep copies retain correct method binding, and unrelated modules remain unchanged. The Vitis AI host remains the required integration environment for comparing CPU operator counts and DPU subgraph counts against the existing baseline.
