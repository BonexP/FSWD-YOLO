# FSWD-YOLO DPU Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a separately trainable FSWD-YOLO DPU candidate that preserves the model's three-branch design identity, removes known target-side split/decode blockers, supports deterministic checkpoint initialization, and can be screened with Vitis AI Inspector before hardware purchase.

**Architecture:** Keep the original `fswd-yolo.yaml` and `C2PSFCA` unchanged. Add DPU-specific modules in a focused module file, a new Hardswish-based YAML, a split-free checkpoint migration path, and a raw Detect-head adapter that leaves DFL/decode/NMS on the host. Extend the existing Inspector tool so it can inspect either a trained checkpoint or an explicitly labelled random-weight YAML candidate, and gate training on Inspector plus parameter/FLOP evidence.

**Tech Stack:** Python 3.8+, PyTorch, Ultralytics model parser, unittest/pytest, thop-based Ultralytics profiling, Vitis AI 3.5 PyTorch NNDCT Inspector.

---

## Preconditions

- Work only in the `components/FSWD-YOLO/` submodule on branch `codex/fswd-onnx-deployment-tools`.
- Preserve `docs/quan_dev/logs.txt`; it is user-owned and untracked.
- Do not modify `ultralytics/cfg/models/11/fswd-yolo.yaml` or the existing `C2PSFCA` class.
- Run PyTorch tests with the activated FSWD-YOLO environment on the model host; every command below uses that environment's `python`. The current Codex desktop runtime has Python 3.12 but no PyTorch or pytest, so desktop verification is limited to dependency-light `unittest`, syntax compilation, and static checks unless the user supplies a model environment.
- Commit and push each coherent submodule change before updating the paper-workspace submodule pointer.

## Task 1: Add Split-Free DPU Building Blocks

**Files:**

- Create: `ultralytics/nn/modules/fswd_dpu.py`
- Create: `tests/test_fswd_dpu_model.py`

- [ ] **Step 1: Write failing shape and gradient tests for `C2PSFCADPU`**

Add tests that import `C2PSFCADPU`, construct `C2PSFCADPU(128, 128, n=1, e=0.5, channel_reduction=4)`, and assert:

```python
x = torch.randn(2, 128, 20, 20, requires_grad=True)
y = module(x)
assert y.shape == x.shape
assert torch.isfinite(y).all()
y.mean().backward()
assert torch.isfinite(x.grad).all()
```

Also register forward hooks on `keep_projection`, `spatial_projection`, and `channel_projection` and assert that all three outputs are `(2, 64, 20, 20)`.

- [ ] **Step 2: Write failing graph-form tests**

Assert that the new module tree contains:

- `nn.Hardswish` and `nn.Hardsigmoid`;
- `nn.AdaptiveAvgPool2d` for the channel branch;
- a depthwise 5x5 convolution in the spatial branch;
- no module whose forward implementation uses `split` or `chunk`.

Run:

```powershell
python -m pytest tests/test_fswd_dpu_model.py -k "c2psfca" -q
```

Expected: collection/import failure because `fswd_dpu.py` does not exist yet.

- [ ] **Step 3: Implement the focused DPU module file**

Create `SplitFreeC2f`, `C3k2DPU`, `C3k2GhostSimAMinnerDPU`, `DPUSpatialAttentionBlock`, `DPUChannelAttentionBlock`, and `C2PSFCADPU`. All inherit `nn.Module` directly or through `SplitFreeC2f`; none inherit the split-based `C2f` implementation.

`SplitFreeC2f` must use two independent 1x1 projections:

```python
y = [self.cv_keep(x), self.cv_process(x)]
y.extend(block(y[-1]) for block in self.m)
return self.cv_fuse(torch.cat(y, dim=1))
```

`C2PSFCADPU` must use three independent 1x1 projections and preserve the approved branch semantics:

```python
keep = self.keep_projection(x)
spatial = self.spatial_blocks(self.spatial_projection(x))
channel = self.channel_blocks(self.channel_projection(x))
return self.fuse(torch.cat((keep, spatial, channel), dim=1))
```

Spatial block:

```python
gate = self.gate(self.pointwise(self.depthwise(x)))
x = x + x * gate
return x + self.ffn(x)
```

Channel block:

```python
gate = self.gate(self.expand(self.activate(self.reduce(self.pool(x)))))
return x * gate
```

Use explicit `nn.Hardswish()` and `nn.Hardsigmoid()` inside DPU attention blocks. Use `max(1, channels // channel_reduction)` for the channel bottleneck.

- [ ] **Step 4: Run the focused tests**

```powershell
python -m pytest tests/test_fswd_dpu_model.py -k "c2psfca or split_free" -q
```

Expected: all focused tests pass.

- [ ] **Step 5: Commit**

```powershell
git add ultralytics/nn/modules/fswd_dpu.py tests/test_fswd_dpu_model.py
git commit -m "feat: add split-free FSWD DPU modules"
```

## Task 2: Register the DPU Model and Isolate Activation State

**Files:**

- Modify: `ultralytics/nn/modules/__init__.py`
- Modify: `ultralytics/nn/tasks.py`
- Create: `ultralytics/cfg/models/11/fswd-yolo-dpu.yaml`
- Modify: `tests/test_fswd_dpu_model.py`

- [ ] **Step 1: Write failing parser and construction tests**

Add tests that verify:

1. `C2PSFCADPU`, `C3k2DPU`, and `C3k2GhostSimAMinnerDPU` are exported from `ultralytics.nn.modules`.
2. `YOLO("ultralytics/cfg/models/11/fswd-yolo-dpu.yaml").model` constructs successfully.
3. The constructed model contains exactly one `C2PSFCADPU`, five `C3k2DPU`, and two `C3k2GhostSimAMinnerDPU` modules.
4. Its convolution activations use Hardswish.
5. Constructing the original `fswd-yolo.yaml` immediately afterward restores the original `Conv.default_act` and produces SiLU activations.

Run:

```powershell
python -m pytest tests/test_fswd_dpu_model.py -k "registration or yaml or activation" -q
```

Expected: failures because the classes and YAML are not registered.

- [ ] **Step 2: Export and register the new classes**

Import the three public DPU model classes from `fswd_dpu.py` in `ultralytics/nn/modules/__init__.py`, add them to `__all__`, and import them in `ultralytics/nn/tasks.py`.

Add all three to `base_modules`. Add `C3k2DPU` and `C2PSFCADPU` to `repeat_modules`, matching their original counterparts. Keep `C3k2GhostSimAMinnerDPU` out of `repeat_modules`, matching `C3k2GhostSimAMinner` so YAML depth repetition remains an outer `nn.Sequential` instead of silently becoming internal repetition.

- [ ] **Step 3: Prevent activation state leakage in `parse_model`**

Preserve `Conv.default_act` before applying a YAML-specific activation, build the model inside `try`, and restore the previous value in `finally`:

```python
previous_default_act = Conv.default_act
try:
    if act:
        Conv.default_act = eval(act)
    # existing layer-construction loop
finally:
    Conv.default_act = previous_default_act
```

The constructed modules retain their own activation instances; unrelated models no longer inherit the preceding YAML's activation setting, including when parsing raises an exception.

- [ ] **Step 4: Add `fswd-yolo-dpu.yaml`**

Copy the original topology and make only these substitutions:

```yaml
activation: torch.nn.Hardswish()
```

- Layers 2 and 4: `C3k2GhostSimAMinnerDPU`
- Layers 6, 8, 13, 16, and 22: `C3k2DPU`
- Layer 10: `C2PSFCADPU`
- Detect and all connection indices: unchanged

- [ ] **Step 5: Run parser regressions**

```powershell
python -m pytest tests/test_fswd_dpu_model.py -k "registration or yaml or activation" -q
python -m pytest tests/test_python.py -k "model" -q
```

Expected: new model tests pass; existing model construction remains green.

- [ ] **Step 6: Commit**

```powershell
git add ultralytics/nn/modules/__init__.py ultralytics/nn/tasks.py ultralytics/cfg/models/11/fswd-yolo-dpu.yaml tests/test_fswd_dpu_model.py
git commit -m "feat: register FSWD DPU model candidate"
```

## Task 3: Implement Exact Split-Free Projection Migration

**Files:**

- Create: `scripts/fswd_dpu_migration.py`
- Modify: `tests/test_fswd_dpu_model.py`

- [ ] **Step 1: Write failing slice-copy parity tests**

For original `C3k2` and new `C3k2DPU` modules with the same dimensions and activation:

1. Seed PyTorch.
2. Copy all same-name compatible parameters.
3. Copy output channels `[0:c]` from original `cv1` into `cv_keep`.
4. Copy output channels `[c:2*c]` into `cv_process`.
5. Compare outputs on the same random tensor with `rtol=0`, `atol=0`.

Repeat for `C3k2GhostSimAMinner` and `C3k2GhostSimAMinnerDPU`.

Run:

```powershell
python -m pytest tests/test_fswd_dpu_model.py -k "migration and parity" -q
```

Expected: failure because migration helpers do not exist.

- [ ] **Step 2: Implement migration primitives**

Add `copy_conv_output_slice(source, destination, start, end)`, `copy_matching_state(source, destination)`, `migrate_split_free_module(source, destination)`, `migrate_c2psfca(source, destination)`, and `migrate_fswd_model(source, destination)`. Use a typed `MigrationSummary` dataclass for module-level copied, sliced, new, unmapped, and unexpected tensor records; return a JSON-serializable dictionary from the model-level function.

`copy_conv_output_slice` must copy convolution weights/bias plus BatchNorm `weight`, `bias`, `running_mean`, `running_var`, and `num_batches_tracked`.

The model-level migration must fail on:

- layer-count or layer-index mismatch;
- unexpected incompatible same-name tensors;
- unsupported source/destination custom-module pairs;
- missing projection slices.

The approved attention redesign is not parity-equivalent, so its report must classify:

- copied compatible tensors;
- projection slices copied;
- newly initialized tensors;
- intentionally unmapped original PSA/FCA tensors;
- unexpected mismatches, which must be empty for success.

- [ ] **Step 3: Add migration accounting tests**

Construct original and DPU YAML models in memory and assert:

- all split-free layers report exact isolated parity;
- the `C2PSFCADPU` row reports three projection slices and new attention parameters;
- no tensor is silently omitted;
- an intentionally corrupted tensor shape raises a migration error.

- [ ] **Step 4: Run tests**

```powershell
python -m pytest tests/test_fswd_dpu_model.py -k "migration" -q
```

Expected: all migration tests pass.

- [ ] **Step 5: Commit**

```powershell
git add scripts/fswd_dpu_migration.py tests/test_fswd_dpu_model.py
git commit -m "feat: migrate FSWD weights into split-free DPU graph"
```

## Task 4: Add Raw Detect Output and Host Decode Parity

**Files:**

- Modify: `ultralytics/nn/modules/head.py`
- Create: `scripts/fswd_dpu_model.py`
- Modify: `tests/test_fswd_dpu_model.py`

- [ ] **Step 1: Write failing raw-head contract tests**

Add tests for a DPU YAML model in eval mode:

- native output remains a decoded tensor plus raw feature maps;
- the deployment adapter returns exactly three tensors;
- shapes are `(1, no, 80, 80)`, `(1, no, 40, 40)`, and `(1, no, 20, 20)` at 640x640;
- the adapter adds no parameters or buffers;
- the original model's raw-output setting is restored after adapter use.

- [ ] **Step 2: Write failing decode parity test**

Decode cloned adapter outputs through the model's existing Detect `_inference` implementation and compare against the native FP32 decoded output:

```python
assert torch.allclose(native, decoded_from_raw, rtol=1e-5, atol=1e-6)
```

Run:

```powershell
python -m pytest tests/test_fswd_dpu_model.py -k "raw_head or decode" -q
```

Expected: failures because the raw-output path and adapter do not exist.

- [ ] **Step 3: Add an explicit raw-output flag to Detect**

Add an instance attribute initialized to `False`, and return the three concatenated feature maps when either training or explicit raw-output mode is active:

```python
if self.training or self.raw_output:
    return x
```

Do not alter the default training or inference return contracts.

- [ ] **Step 4: Implement the deployment adapter**

In `scripts/fswd_dpu_model.py`, add `RawDetectHeadAdapter`, `find_detect_head(model)`, `decode_raw_predictions(model, raw_outputs)`, and `compare_decode_parity(model, input_tensor, rtol=1e-5, atol=1e-6)`.

The adapter must use `try/finally` to restore `Detect.raw_output`, return an immutable tuple of tensors, and never run DFL/decode/sigmoid/NMS inside its forward path.

- [ ] **Step 5: Run raw-head tests**

```powershell
python -m pytest tests/test_fswd_dpu_model.py -k "raw_head or decode" -q
```

Expected: all raw-output and parity tests pass.

- [ ] **Step 6: Commit**

```powershell
git add ultralytics/nn/modules/head.py scripts/fswd_dpu_model.py tests/test_fswd_dpu_model.py
git commit -m "feat: expose DPU raw detection outputs"
```

## Task 5: Extend Inspector for YAML Candidates and Raw Head Inspection

**Files:**

- Modify: `scripts/fswd_deploy_common.py`
- Modify: `scripts/inspect_fswd_vitis.py`
- Modify: `tests/test_fswd_deployment_tools.py`

- [ ] **Step 1: Write failing CLI source-selection tests**

Update tests to require exactly one of:

```text
--weights PATH.pt
--model-config PATH.yaml
```

Add `--raw-detect-output` as a boolean flag. Assert:

- neither source is rejected by argparse;
- both sources together are rejected;
- YAML suffix/path validation is enforced;
- activation experiments are rejected for YAML candidates because the DPU YAML already defines activation semantics;
- raw-head mode is accepted for either source.

- [ ] **Step 2: Write failing manifest tests**

For checkpoint source, expect:

```json
"model_source": {
  "kind": "trained_checkpoint",
  "path": "/workspace/best.pt",
  "sha256": "computed from /workspace/best.pt",
  "weight_state": "trained"
}
```

For YAML source, expect:

```json
"model_source": {
  "kind": "model_config_candidate",
  "path": "/workspace/FSWD-YOLO/ultralytics/cfg/models/11/fswd-yolo-dpu.yaml",
  "sha256": "computed from fswd-yolo-dpu.yaml",
  "weight_state": "random_initialization"
}
```

The manifest must also record `output_contract` as `decoded_predictions` or `raw_detect_feature_maps`.

Run:

```powershell
python -m pytest tests/test_fswd_deployment_tools.py -k "inspector and (source or manifest or raw)" -q
```

Expected: failures against the current weights-only CLI.

- [ ] **Step 3: Add common model-config validation**

Add `validate_model_config(path: Path) -> Path` beside `validate_weights`. It must require an existing regular `.yaml` or `.yml` file and must not install dependencies.

- [ ] **Step 4: Refactor Inspector model loading**

Use a mutually exclusive argparse group, resolve the selected source, and load either:

```python
model = YOLO(str(weights)).model.float().eval().cpu()
```

or:

```python
model = YOLO(str(model_config)).model.float().eval().cpu()
```

Wrap with `RawDetectHeadAdapter` only when `--raw-detect-output` is set.

Replace single-tensor preparation comparison with a sequence-aware comparison that records every output shape and aggregate maximum/mean absolute errors. Existing single decoded tensor manifests must remain readable and retain their current semantic-equivalence enforcement.

- [ ] **Step 5: Run deployment-tool tests**

```powershell
python -m pytest tests/test_fswd_deployment_tools.py -q
```

Expected: all existing and new deployment-tool tests pass.

- [ ] **Step 6: Commit**

```powershell
git add scripts/fswd_deploy_common.py scripts/inspect_fswd_vitis.py tests/test_fswd_deployment_tools.py
git commit -m "feat: inspect FSWD YAML candidates and raw heads"
```

## Task 6: Add the Checkpoint Initialization CLI

**Files:**

- Create: `scripts/initialize_fswd_dpu.py`
- Modify: `tests/test_fswd_deployment_tools.py`

- [ ] **Step 1: Write failing CLI tests**

Cover:

- required `--source-checkpoint`, `--model-config`, and `--output`;
- default report path `<output>.migration.json`;
- positive `--imgsz` validation;
- overwrite protection for both checkpoint and report;
- missing dependency errors that state the tool checks only and installs nothing.

- [ ] **Step 2: Write failing checkpoint round-trip test**

Use a temporary source checkpoint created from the original YAML model. Run the CLI's internal `run()` with the DPU YAML, then assert:

- the source checkpoint SHA256 is unchanged;
- output checkpoint and JSON report exist;
- `YOLO(output_checkpoint)` reloads successfully;
- output model contains DPU module classes;
- report status is `ok` and has no unexpected mismatches;
- split-free parity evidence is present;
- attention redesign tensors are explicitly classified as new/unmapped.

- [ ] **Step 3: Implement the CLI**

Use only repository imports and active-environment dependencies. Set `YOLO_AUTOINSTALL=false`. Preserve useful source checkpoint metadata but replace model state with the migrated DPU model; clear optimizer/EMA training state that cannot be valid for a changed graph.

The report must include:

- arguments, timestamps, Git metadata, runtime and package versions;
- source and output SHA256 values;
- source/destination model YAML paths;
- copied, sliced, new, intentionally unmapped, and unexpected tensor counts;
- isolated split-free parity results;
- full-model parameter/FLOP comparison;
- raw-head decode parity result;
- final status/error.

- [ ] **Step 4: Run CLI tests**

```powershell
python -m pytest tests/test_fswd_deployment_tools.py -k "initialize_fswd_dpu or migration_cli" -q
```

Expected: all initialization CLI tests pass.

- [ ] **Step 5: Commit**

```powershell
git add scripts/initialize_fswd_dpu.py tests/test_fswd_deployment_tools.py
git commit -m "feat: initialize FSWD DPU training checkpoints"
```

## Task 7: Add Resource Gates and Candidate Summary

**Files:**

- Create: `scripts/profile_fswd_dpu_candidate.py`
- Modify: `tests/test_fswd_deployment_tools.py`

- [ ] **Step 1: Write failing budget tests**

Build original and DPU YAML models at 640x640 and assert the report calculates:

```text
parameter_ratio = dpu_parameters / original_parameters
flop_ratio = dpu_flops / original_flops
parameter_gate = parameter_ratio <= 1.10
flop_gate = dpu_flops <= original_flops
```

Add a failure fixture whose synthetic metrics exceed each threshold.

- [ ] **Step 2: Implement the profiler CLI**

CLI:

```text
python scripts/profile_fswd_dpu_candidate.py \
  --baseline-config ultralytics/cfg/models/11/fswd-yolo.yaml \
  --candidate-config ultralytics/cfg/models/11/fswd-yolo-dpu.yaml \
  --imgsz 640 \
  --output fswd_dpu_profile.json \
  --overwrite
```

Use the repository's established model-info/profile path, not hand-written FLOP formulas. Record model-level totals and per-module-type parameter deltas. Exit with code 2 when either gate fails while still writing the report.

- [ ] **Step 3: Run tests and a local profile**

```powershell
python -m pytest tests/test_fswd_deployment_tools.py -k "profile_fswd_dpu" -q
python scripts/profile_fswd_dpu_candidate.py --baseline-config ultralytics/cfg/models/11/fswd-yolo.yaml --candidate-config ultralytics/cfg/models/11/fswd-yolo-dpu.yaml --imgsz 640 --output tmp/fswd_dpu_profile.json --overwrite
```

Expected: tests pass; the command writes a JSON report and clearly prints pass/fail for both gates.

- [ ] **Step 4: Commit**

```powershell
git add scripts/profile_fswd_dpu_candidate.py tests/test_fswd_deployment_tools.py
git commit -m "feat: gate FSWD DPU candidate resources"
```

## Task 8: Document and Verify the End-to-End Screening Workflow

**Files:**

- Modify: `docs/quan_dev/README.md`
- Modify: `docs/superpowers/specs/2026-07-24-fswd-yolo-dpu-model-design.md` only if implementation details reveal a factual inconsistency; otherwise leave it unchanged.

- [ ] **Step 1: Document the pre-training local workflow**

Add commands for:

1. profiling original vs DPU YAML;
2. initializing a DPU checkpoint from `best.pt`;
3. verifying raw-head decode parity;
4. copying the repository and candidate artifacts to the Vitis AI host.

State explicitly that the initialized checkpoint is not accuracy-valid until retrained.

- [ ] **Step 2: Document Vitis AI candidate inspection commands**

Random-weight graph screening before training:

```bash
python scripts/inspect_fswd_vitis.py \
  --model-config ultralytics/cfg/models/11/fswd-yolo-dpu.yaml \
  --target DPUCZDX8G_ISA1_B4096 \
  --raw-detect-output \
  --output-dir /workspace/inspect_fswd_dpu_candidate \
  --overwrite
```

Initialized or later trained checkpoint screening:

```bash
python scripts/inspect_fswd_vitis.py \
  --weights /workspace/fswd-yolo-dpu-init.pt \
  --target DPUCZDX8G_ISA1_B4096 \
  --raw-detect-output \
  --output-dir /workspace/inspect_fswd_dpu_checkpoint \
  --overwrite
```

Explain that `DPUCZDX8G_ISA1_B4096` remains a screening target only; final quantization/compile must use the selected board's `arch.json` or fingerprint.

- [ ] **Step 3: Define the graph gate command**

After Inspector completes:

```bash
python -m json.tool /workspace/inspect_fswd_dpu_candidate/inspection_manifest.json
grep -n "assigned to CPU\|can't be converted to XIR" /workspace/inspect_fswd_dpu_candidate/inspect_*.txt
```

Pass criteria:

- zero direct unsupported operators in backbone, neck, `C2PSFCADPU`, convolutional Detect head, and raw outputs;
- CPU work is limited to the explicitly external host decode/NMS pipeline, which is absent from the inspected graph;
- official Inspector output still reports at least one DPU subgraph;
- parameter and FLOP gates pass;
- evidence comes from a clean submodule worktree.

- [ ] **Step 4: Run the complete local regression suite**

```powershell
python -m pytest tests/test_fswd_dpu_model.py tests/test_fswd_deployment_tools.py -q
python -m compileall scripts ultralytics/nn/modules/fswd_dpu.py
git diff --check
```

Expected: all tests pass, compilation succeeds, and `git diff --check` prints nothing.

- [ ] **Step 5: Commit documentation**

```powershell
git add docs/quan_dev/README.md
git commit -m "docs: add FSWD DPU screening workflow"
```

- [ ] **Step 6: Push the submodule branch**

```powershell
git push origin codex/fswd-onnx-deployment-tools
```

- [ ] **Step 7: Update the paper-workspace submodule pointer separately**

From `C:\Users\iamho\Documents\Projects\FSWD-paper-draft`:

```powershell
git add components/FSWD-YOLO
git commit -m "chore: update FSWD-YOLO DPU tooling pointer"
git push origin codex/revision-two-track-cleanup
```

Do not stage or commit the unrelated manuscript files or `docs/quan_dev/logs.txt`.

## Final Verification Handoff

The implementation is locally complete only after the local regression, compile, diff, and resource checks pass. DPU compatibility remains `Inspector verified` only after the user runs Task 8 on the Vitis AI 3.5 host and returns the generated `inspection_manifest.json` plus `inspect_*.txt`. Formal retraining starts only after those graph/resource gates pass and the user approves the training protocol.
