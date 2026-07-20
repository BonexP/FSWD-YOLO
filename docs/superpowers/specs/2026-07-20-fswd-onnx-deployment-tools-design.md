# FSWD-YOLO ONNX Deployment Tools Design

## Goal

Provide a reproducible, target-neutral toolchain for exporting an FSWD-YOLO checkpoint to fixed-shape FP32 ONNX, validating the exported graph, optionally comparing PyTorch and ONNX Runtime outputs, and inspecting the PyTorch graph with Vitis AI 3.5.

The tools must never install, upgrade, or remove dependencies. They must preserve the Vitis AI Conda environment and fail with actionable diagnostics when a required package is unavailable or incompatible.

## Scope

The implementation adds:

- `scripts/fswd_deploy_common.py` for dependency checks, path safety, version and Git metadata, SHA256 hashes, JSON reports, and operator summaries.
- `scripts/export_fswd_onnx.py` for ONNX export, structural validation, operator inventory, and optional ONNX Runtime parity checks.
- `scripts/inspect_fswd_vitis.py` for Vitis AI NNDCT Inspector execution and inspection manifests.
- `tests/test_fswd_deployment_tools.py` for dependency-free unit coverage and environment-gated integration coverage.
- `docs/quan_dev/README.md` for the two-host workflow, environment protection rules, commands, and report interpretation.

The implementation does not quantize or compile a model, select an FPGA board, choose an Inspector target, or alter Ultralytics' shared exporter behavior.

## Architecture

The tools are separate command-line programs because ONNX export and Vitis inspection often run in different Python environments. Shared behavior remains in a standard-library-first helper module so it can be tested without importing PyTorch, ONNX, ONNX Runtime, Ultralytics, or NNDCT.

All framework-specific imports are lazy. Each command performs dependency preflight checks before importing its framework stack. The checks use import metadata only and never call `pip`, `conda`, Ultralytics requirement installers, or subprocess-based installers.

## ONNX Export CLI

The primary command is:

```bash
python scripts/export_fswd_onnx.py \
  --weights /path/to/best.pt \
  --output /path/to/fswd-yolo.onnx \
  --imgsz 640 \
  --opset 13
```

The exporter uses these deployment-oriented invariants:

- batch size: `1`
- input shape: fixed square image controlled by `--imgsz`, default `640`
- precision: FP32
- dynamic axes: disabled
- NMS: excluded
- simplification: disabled
- ONNX opset: default `13`

The command validates the weights before loading the model and rejects an existing output unless `--overwrite` is present. Ultralytics may initially write beside the checkpoint; the command relocates the result to the requested output only after export succeeds.

After export, the command loads the ONNX model and runs `onnx.checker.check_model`. It records input and output names, element types and shapes, an operator histogram, and counts for review-sensitive operations:

```text
ReduceMean, ReduceSum, Pow, Div, MatMul, Softmax, Expand, Transpose
```

These operations are marked for target-specific review, not declared unsupported.

The command writes `<output>.report.json` atomically. The report includes command arguments, status, errors, timestamps, source and output SHA256 hashes, Git commit, runtime versions, ONNX graph information, and optional parity results.

## Runtime Parity Check

`--verify-runtime` enables a deterministic PyTorch versus ONNX Runtime comparison. The command generates a random FP32 tensor using a fixed seed and runs the loaded FSWD-YOLO model and ONNX Runtime CPU provider with the same tensor.

The PyTorch result normalizer selects the exported prediction tensor from the model's inference return structure. The comparison requires equal shapes and records:

- maximum absolute error
- mean absolute error
- configured `rtol` and `atol`
- `numpy.allclose` result

Default tolerances are `rtol=1e-4` and `atol=1e-5`. A failed parity check preserves the ONNX file and report but exits nonzero. `onnxruntime` is required only when this option is enabled.

## Vitis Inspector CLI

The primary command is:

```bash
python scripts/inspect_fswd_vitis.py \
  --weights /path/to/best.pt \
  --target ACTUAL_TARGET \
  --output-dir /path/to/inspect_fswd
```

`--target` is required. The tool does not provide a guessed DPU or board default. The user must obtain a target name or fingerprint that matches the intended hardware.

The command loads the checkpoint with local Ultralytics code, extracts the PyTorch model, converts it to FP32 evaluation mode on CPU, constructs a fixed input tensor, and calls `pytorch_nndct.apis.Inspector.inspect`.

It supports `--verbose-level` and `--image-format` values accepted by Inspector. It writes `inspection_manifest.json` atomically with the target, checkpoint hash, Git commit, runtime versions, input shape, Inspector settings, status, and error details. An existing manifest requires `--overwrite`; other Inspector output files are left intact.

## Dependency Policy

Dependency failures identify:

- the missing or incompatible package
- the active Python executable
- the operation that requires it
- a manual remediation hint

No tool performs installation. Documentation specifically warns against Conda operations that replace the PyTorch build bundled with Vitis AI. Suggested optional packages use pinned `python -m pip install --no-deps ...` commands and require an immediate `import pytorch_nndct` regression check.

## Failure Behavior

Expected validation and environment failures use concise stderr messages and nonzero exit codes. Unexpected exceptions retain their original traceback unless the CLI can add precise context without hiding the cause.

The tools follow these artifact rules:

- Never delete an existing checkpoint, ONNX model, report, or Inspector output.
- Never overwrite an output without explicit authorization.
- Preserve a successfully exported ONNX model when later checker or parity validation fails.
- Write failure status to a report or manifest when its destination can be created safely.
- Use temporary files plus atomic replacement for JSON metadata.

## Testing

Dependency-free unit tests cover:

- export defaults and Inspector's required `--target`
- missing dependency diagnostics without installation side effects
- checkpoint suffix and existence validation
- output overwrite protection
- atomic JSON report writing
- SHA256 calculation
- operator histogram and review-sensitive operation summaries
- JSON-safe version and Git metadata

Environment-gated tests cover ONNX model inspection when ONNX is installed. A small FSWD-YOLO export smoke test is marked as integration/slow and runs only when PyTorch, ONNX, and the local Ultralytics package are available.

Real `best.pt` export, ONNX Runtime parity, and Vitis Inspector runs remain documented host smoke tests because checkpoints, Vitis NNDCT kernels, and board targets are not available in default CI.

## Repository and Delivery

All source, tests, and deployment documentation are committed inside the FSWD-YOLO repository on branch `codex/fswd-onnx-deployment-tools`. The user's existing `docs/quan_dev/logs.txt` is diagnostic input and is not included in commits unless explicitly requested.

After tests and static checks pass, the branch is pushed to `origin`. The paper workspace submodule pointer is not updated or committed because the requested deliverable is the independently usable FSWD-YOLO branch; the pointer can be updated only after the pushed submodule commit is intentionally adopted by the paper workspace.
