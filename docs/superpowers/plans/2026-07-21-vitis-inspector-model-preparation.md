# Vitis Inspector Model Preparation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prepare an in-memory FSWD-YOLO model for Vitis AI inspection without changing checkpoint semantics.

**Architecture:** Add a dependency-injected transformation helper to the existing Inspector CLI, then compare deterministic raw predictions before and after transformation. Record every change and parity metric in the inspection manifest.

**Tech Stack:** Python 3.8, PyTorch 1.13, Ultralytics model modules, Vitis AI 3.5 NNDCT, unittest.

---

### Task 1: Dependency-Free Model Preparation

**Files:**
- Modify: `scripts/inspect_fswd_vitis.py`
- Modify: `tests/test_fswd_deployment_tools.py`

- [ ] **Step 1: Write failing preparation tests**

Add fake SiLU, C2f, and unrelated modules. Assert that the helper disables only in-place SiLU, binds `forward_split` only on C2f instances, reports exact counts, and preserves method binding after `copy.deepcopy`.

- [ ] **Step 2: Verify RED**

Run: `python tests/test_fswd_deployment_tools.py InspectorCliTests -v`

Expected: failure because `prepare_model_for_vitis_inspection` does not exist.

- [ ] **Step 3: Implement the preparation helper**

Implement `prepare_model_for_vitis_inspection(model, silu_class, c2f_class) -> Dict[str, int]` using `model.modules()`, `module.inplace = False`, and `module.forward = module.forward_split`.

- [ ] **Step 4: Verify GREEN**

Run: `python tests/test_fswd_deployment_tools.py InspectorCliTests -v`

Expected: all Inspector CLI tests pass without importing PyTorch or NNDCT.

### Task 2: Runtime Equivalence Gate and Manifest

**Files:**
- Modify: `scripts/inspect_fswd_vitis.py`
- Modify: `docs/quan_dev/README.md`

- [ ] **Step 1: Add deterministic prediction comparison**

Run the loaded evaluation model on the Inspector dummy tensor, prepare the model, run it again, normalize the first tensor-like prediction, and require shape equality plus `torch.allclose(rtol=1e-5, atol=1e-6)`.

- [ ] **Step 2: Record evidence**

Write preparation counts, shapes, tolerances, maximum error, mean error, and `allclose` to `inspection_manifest.json` for both success and failure paths.

- [ ] **Step 3: Document the second-pass workflow**

Explain the transformations, semantic gate, expected reduction of `aten::silu_` and `nndct_strided_slice`, and the commands for comparing inspection reports.

- [ ] **Step 4: Run complete verification**

Run:

```bash
python tests/test_fswd_deployment_tools.py -v
python -m py_compile scripts/fswd_deploy_common.py scripts/export_fswd_onnx.py scripts/inspect_fswd_vitis.py tests/test_fswd_deployment_tools.py
python scripts/inspect_fswd_vitis.py --help
git diff --check
```

Expected: all dependency-free tests pass, the ONNX integration test is either green or explicitly skipped, and all static checks exit zero.

- [ ] **Step 5: Commit and push**

Commit only the FSWD-YOLO submodule files and push `codex/fswd-onnx-deployment-tools`. Do not add `docs/quan_dev/logs.txt` or update the paper-workspace submodule pointer.
