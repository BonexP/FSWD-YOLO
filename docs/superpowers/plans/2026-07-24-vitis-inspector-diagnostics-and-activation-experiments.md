# Vitis Inspector Diagnostics and Activation Experiments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add trustworthy Inspector report summaries and explicit Hardswish/Hardsigmoid graph experiments while keeping the default command an exact audit of the original checkpoint.

**Architecture:** Put dependency-free report parsing and generated-report discovery in `scripts/fswd_vitis_report.py`. Keep model preparation and activation replacement in `scripts/inspect_fswd_vitis.py`, with strict parity required only for the default audit mode. The Inspector CLI will attach the parsed summary to its existing atomic manifest.

**Tech Stack:** Python 3.8, standard library, PyTorch 1.13, Ultralytics source checkout, Vitis AI 3.5 NNDCT, `unittest`.

---

### Task 1: Dependency-Free Inspector Report Parser

**Files:**
- Create: `scripts/fswd_vitis_report.py`
- Modify: `tests/test_fswd_deployment_tools.py`

- [ ] **Step 1: Write failing parser tests**

Add `VitisReportTests` with a representative hardware-constraint table deliberately repeated three times to verify protection against copied or concatenated report sections. Assert that `summarize_inspector_report_text()` returns:

```python
{
    "parsed_row_count": 9,
    "unique_row_count": 3,
    "repeated_row_count": 6,
    "unique_node_count": 3,
    "category_counts": {"direct_unsupported": 1, "downstream_cpu": 1, "other_cpu_constraint": 1},
    "operators_by_category": {
        "direct_unsupported": {"aten::silu": 1},
        "downstream_cpu": {"nndct_permute": 1},
        "other_cpu_constraint": {"nndct_custom": 1},
    },
    "requires_cpu_fallback": True,
}
```

Also assert that repeated rows produce one direct-blocker example, a valid empty hardware-constraint table produces zero findings, and unrelated non-empty text raises `InspectorReportError`.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
python tests/test_fswd_deployment_tools.py VitisReportTests -v
```

Expected: import or attribute failure because `scripts.fswd_vitis_report` does not exist.

- [ ] **Step 3: Implement the minimal parser**

Create:

```python
class InspectorReportError(RuntimeError):
    pass


def classify_constraint_reason(reason: str) -> str:
    if any(marker in reason for marker in DIRECT_MARKERS):
        return "direct_unsupported"
    if any(marker in reason for marker in DOWNSTREAM_MARKERS):
        return "downstream_cpu"
    return "other_cpu_constraint"


def summarize_inspector_report_text(text: str) -> Dict[str, Any]:
    rows = parse_constraint_rows(text)
    if not rows and "hardware constraints" not in text.lower():
        raise InspectorReportError("Inspector report does not contain a hardware constraints table")
    return summarize_constraint_rows(rows)
```

Use a compiled regular expression for node, operator, and reason fields separated by two or more spaces. Deduplicate exact `(node, operator, reason)` tuples, sort all dictionary keys, and retain at most three representative examples per direct operator.

- [ ] **Step 4: Run parser tests and verify GREEN**

Run the same focused test command. Expected: all `VitisReportTests` pass.

### Task 2: Generated Report Discovery

**Files:**
- Modify: `scripts/fswd_vitis_report.py`
- Modify: `tests/test_fswd_deployment_tools.py`

- [ ] **Step 1: Write failing discovery tests**

Use temporary directories to cover:

- a newly created `inspect_TARGET.txt`
- an existing report whose `(mtime_ns, size)` signature changes
- no changed reports
- two changed reports

The latter two cases must raise `InspectorReportError` with actionable messages.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
python tests/test_fswd_deployment_tools.py VitisReportTests -v
```

Expected: failure because `snapshot_report_signatures()` and `find_generated_report()` are missing.

- [ ] **Step 3: Implement signature-based discovery**

Implement:

```python
def snapshot_report_signatures(output_dir: Path) -> Dict[Path, Tuple[int, int]]:
    return {
        path.resolve(): (path.stat().st_mtime_ns, path.stat().st_size)
        for path in output_dir.glob("inspect_*.txt")
        if path.is_file()
    }


def find_generated_report(output_dir: Path, before: Mapping[Path, Tuple[int, int]]) -> Path:
    after = snapshot_report_signatures(output_dir)
    changed = sorted(path for path, signature in after.items() if before.get(path) != signature)
    if not changed:
        raise InspectorReportError("Inspector did not generate or update an inspect_*.txt report")
    if len(changed) > 1:
        raise InspectorReportError("Inspector generated or updated multiple inspect_*.txt reports")
    return changed[0]
```

- [ ] **Step 4: Run focused tests and verify GREEN**

Expected: all report tests pass.

### Task 3: Exact Audit Preparation and Activation Experiment API

**Files:**
- Modify: `scripts/inspect_fswd_vitis.py`
- Modify: `tests/test_fswd_deployment_tools.py`

- [ ] **Step 1: Replace the obsolete preparation test**

Change the current C2f test to assert that preparation only disables in-place SiLU and always reports `c2f_forward_split_enabled: 0`. Assert that a fake C2f instance continues to use its original `forward()`.

Add tests for:

```python
apply_activation_experiment(model, FakeSiLU, "hardswish", hard_forward)
```

The helper must bind `hard_forward` to every unique fake SiLU module, report the replacement count, leave unrelated modules unchanged, and preserve method binding after `copy.deepcopy`.

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
python tests/test_fswd_deployment_tools.py InspectorCliTests -v
```

Expected: the old preparation assertion fails and the activation helper is missing.

- [ ] **Step 3: Implement exact preparation and method binding**

Change preparation to:

```python
def prepare_model_for_vitis_inspection(model: Any, silu_class: Any) -> Dict[str, int]:
    changes = {"silu_inplace_disabled": 0, "c2f_forward_split_enabled": 0}
    for module in model.modules():
        if isinstance(module, silu_class) and getattr(module, "inplace", False):
            module.inplace = False
            changes["silu_inplace_disabled"] += 1
    return changes
```

Implement activation binding with `types.MethodType`. Reject `none` in the replacement helper so the caller cannot accidentally count an unchanged model as an experiment.

- [ ] **Step 4: Add mode and comparison-gate tests**

Test a helper that returns mode metadata for `none`, `hardswish`, and `hardsigmoid`. Test that shape mismatch always raises `ModelPreparationError`, while an allclose mismatch raises only when `semantic_equivalence_required` is true.

- [ ] **Step 5: Implement mode metadata and comparison gate**

Return these stable fields:

```python
{
    "mode": "original_model_audit",
    "activation_experiment": "none",
    "semantic_equivalence_required": True,
    "checkpoint_modified": False,
}
```

or the corresponding `deployment_activation_experiment` payload with `semantic_equivalence_required: False`.

- [ ] **Step 6: Run focused tests and verify GREEN**

Expected: all `InspectorCliTests` pass.

### Task 4: Integrate CLI Modes and Manifest Summary

**Files:**
- Modify: `scripts/inspect_fswd_vitis.py`
- Modify: `tests/test_fswd_deployment_tools.py`

- [ ] **Step 1: Write failing CLI-default tests**

Extend the parser-default test to assert:

```python
self.assertEqual(args.activation_experiment, "none")
```

Parse both hard-activation choices and assert invalid values are rejected by argparse.

- [ ] **Step 2: Run focused tests and verify RED**

Expected: `Namespace` has no `activation_experiment`.

- [ ] **Step 3: Add the CLI option and runtime integration**

Add:

```python
parser.add_argument(
    "--activation-experiment",
    choices=("none", "hardswish", "hardsigmoid"),
    default="none",
)
```

In `run()`:

1. snapshot report signatures before Inspector
2. evaluate baseline predictions
3. apply exact preparation
4. bind `torch.nn.functional.hardswish` or `hardsigmoid` only when selected
5. evaluate and record comparison metrics
6. enforce the mode-specific comparison gate
7. print the non-equivalent experiment warning when applicable
8. invoke Inspector
9. locate and parse the generated report
10. write `inspection_summary` before setting `status: ok`

Remove the unused `C2f` import.

- [ ] **Step 4: Add a pure manifest-payload test**

Exercise the mode metadata and a synthetic report summary together, asserting the resulting keys distinguish execution success, semantic equivalence, and CPU fallback.

- [ ] **Step 5: Run Inspector and full unit tests**

Run:

```bash
python tests/test_fswd_deployment_tools.py InspectorCliTests VitisReportTests -v
python tests/test_fswd_deployment_tools.py -v
```

Expected: all dependency-free tests pass; ONNX integration is green or explicitly skipped when ONNX is unavailable.

### Task 5: Documentation and Verification

**Files:**
- Modify: `docs/quan_dev/README.md`

- [ ] **Step 1: Update the Vitis workflow**

Document that `status: ok` is execution status, explain `inspection_summary`, and remove the claim that `forward_split()` reduces slices. Add three separate commands for:

```bash
--activation-experiment none
--activation-experiment hardswish
--activation-experiment hardsigmoid
```

State that the hard-activation runs are graph experiments requiring retraining or fine-tuning before accuracy claims.

- [ ] **Step 2: Run final local verification**

Run:

```bash
python tests/test_fswd_deployment_tools.py -v
python -m py_compile scripts/fswd_deploy_common.py scripts/fswd_vitis_report.py scripts/export_fswd_onnx.py scripts/inspect_fswd_vitis.py tests/test_fswd_deployment_tools.py
python scripts/inspect_fswd_vitis.py --help
git diff --check
```

Expected: tests pass, the ONNX test is green or explicitly skipped, compilation and help exit zero, and `git diff --check` is clean.

- [ ] **Step 3: Commit and push**

Stage only the plan, parser, Inspector CLI, tests, and deployment README. Do not add `docs/quan_dev/logs.txt`. Commit and push `codex/fswd-onnx-deployment-tools`, then verify the local and remote commit IDs match.
