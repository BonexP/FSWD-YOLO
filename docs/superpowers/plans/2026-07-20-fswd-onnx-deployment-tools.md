# FSWD-YOLO ONNX Deployment Tools Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build tested, non-installing CLI tools for fixed-shape FSWD-YOLO ONNX export, graph reporting, optional runtime parity checks, and target-neutral Vitis AI inspection.

**Architecture:** Keep dependency-free validation and reporting in `scripts/fswd_deploy_common.py`. Put framework imports behind explicit preflight checks in two separate CLIs so the ONNX host and Vitis AI host remain independent and the Vitis PyTorch ABI is never modified by the tools.

**Tech Stack:** Python 3.8+, standard library, Ultralytics YOLO, PyTorch, ONNX, optional ONNX Runtime, optional Vitis AI `pytorch_nndct`, unittest/pytest-compatible tests.

---

## File Structure

- Create `scripts/fswd_deploy_common.py`: dependency preflight, file validation, overwrite protection, SHA256, Git/runtime metadata, operator summaries, atomic JSON writes.
- Create `scripts/export_fswd_onnx.py`: export CLI, ONNX structural checks, report generation, optional PyTorch/ONNX Runtime parity.
- Create `scripts/inspect_fswd_vitis.py`: required-target Inspector CLI and inspection manifest.
- Create `tests/test_fswd_deployment_tools.py`: standard-library unit tests runnable directly without importing `tests/__init__.py`.
- Create `docs/quan_dev/README.md`: two-host workflow, dependency safety, commands, reports, and smoke tests.

### Task 1: Dependency-Free Shared Utilities

**Files:**
- Create: `scripts/fswd_deploy_common.py`
- Create: `tests/test_fswd_deployment_tools.py`

- [ ] **Step 1: Write failing tests for path validation, overwrite protection, hashing, reports, and operator summaries**

```python
class CommonUtilitiesTests(unittest.TestCase):
    def test_validate_weights_requires_existing_pt_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.bin"
            path.write_bytes(b"weights")
            with self.assertRaisesRegex(ValueError, "must use the .pt suffix"):
                common.validate_weights(path)

    def test_prepare_output_rejects_existing_file_without_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "model.onnx"
            output.write_bytes(b"existing")
            with self.assertRaisesRegex(FileExistsError, "--overwrite"):
                common.prepare_output(output, overwrite=False)

    def test_sha256_file_is_stable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.bin"
            path.write_bytes(b"fswd")
            self.assertEqual(common.sha256_file(path), hashlib.sha256(b"fswd").hexdigest())

    def test_write_json_atomic_replaces_complete_document(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            common.write_json_atomic(path, {"status": "ok"})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"status": "ok"})

    def test_summarize_operators_counts_review_operations(self):
        summary = common.summarize_operators(["Conv", "Div", "Conv", "MatMul", "Transpose"])
        self.assertEqual(summary["histogram"], {"Conv": 2, "Div": 1, "MatMul": 1, "Transpose": 1})
        self.assertEqual(summary["review_operations"], {"Div": 1, "MatMul": 1, "Transpose": 1})
```

- [ ] **Step 2: Run the focused test file and verify RED**

Run:

```bash
python tests/test_fswd_deployment_tools.py CommonUtilitiesTests -v
```

Expected: import failure because `scripts.fswd_deploy_common` does not exist.

- [ ] **Step 3: Implement the standard-library helpers**

Implement `REVIEW_OPERATIONS` as the exact frozenset shown in the test and add the following public interfaces with the stated signatures: `DependencyError(RuntimeError)`, `require_modules(requirements: Mapping[str, str], operation: str) -> Dict[str, str]`, `validate_weights(path: Path) -> Path`, `prepare_output(path: Path, overwrite: bool, expected_suffix: str = ".onnx") -> Path`, `sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str`, `git_metadata(repo_root: Path) -> Dict[str, Any]`, `package_versions(package_names: Iterable[str]) -> Dict[str, Optional[str]]`, `summarize_operators(operator_types: Iterable[str]) -> Dict[str, Dict[str, int]]`, `write_json_atomic(path: Path, payload: Mapping[str, Any], overwrite: bool = True) -> None`, and `utc_now() -> str`.

`require_modules` uses `importlib.util.find_spec` and `importlib.metadata.version`; it raises `DependencyError` containing the active `sys.executable`, operation, missing package, and manual hint. It never invokes a package manager.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run:

```bash
python tests/test_fswd_deployment_tools.py CommonUtilitiesTests -v
```

Expected: all `CommonUtilitiesTests` pass.

- [ ] **Step 5: Commit shared helpers and tests**

```bash
git add scripts/fswd_deploy_common.py tests/test_fswd_deployment_tools.py
git commit -m "feat: add FSWD deployment utility primitives"
```

### Task 2: Fixed-Shape ONNX Export and Report CLI

**Files:**
- Create: `scripts/export_fswd_onnx.py`
- Modify: `tests/test_fswd_deployment_tools.py`

- [ ] **Step 1: Write failing tests for CLI defaults, dependency selection, and report paths**

```python
class ExportCliTests(unittest.TestCase):
    def test_export_defaults_are_fpga_review_friendly(self):
        parser = export_cli.build_parser()
        args = parser.parse_args(["--weights", "best.pt", "--output", "fswd.onnx"])
        self.assertEqual(args.imgsz, 640)
        self.assertEqual(args.opset, 13)
        self.assertFalse(args.verify_runtime)
        self.assertFalse(args.overwrite)

    def test_export_requirements_add_onnxruntime_only_for_parity(self):
        self.assertEqual(export_cli.export_requirements(False), {"torch": "PyTorch", "onnx": "ONNX", "ultralytics": "Ultralytics"})
        self.assertIn("onnxruntime", export_cli.export_requirements(True))

    def test_report_path_appends_report_json(self):
        self.assertEqual(export_cli.report_path(Path("model.onnx")), Path("model.onnx.report.json"))
```

- [ ] **Step 2: Run export tests and verify RED**

Run:

```bash
python tests/test_fswd_deployment_tools.py ExportCliTests -v
```

Expected: import failure because `scripts.export_fswd_onnx` does not exist.

- [ ] **Step 3: Implement the export CLI**

Implement these public interfaces with consistent signatures: `build_parser() -> argparse.ArgumentParser`, `export_requirements(verify_runtime: bool) -> Dict[str, str]`, `report_path(output: Path) -> Path`, `onnx_graph_report(model: Any, onnx_module: Any) -> Dict[str, Any]`, `normalize_torch_prediction(output: Any) -> Any`, `run_runtime_parity(yolo: Any, output: Path, imgsz: int, seed: int, rtol: float, atol: float) -> Dict[str, Any]`, `run(args: argparse.Namespace) -> int`, and `main() -> int`.

`run` must:

1. Validate `.pt` input and `.onnx` output before framework imports.
2. Preflight `torch`, `onnx`, `ultralytics`, and optional `onnxruntime`.
3. Set `YOLO_AUTOINSTALL=false` before importing Ultralytics.
4. Load `YOLO(weights)` and call `export(format="onnx", imgsz=args.imgsz, batch=1, dynamic=False, nms=False, simplify=False, half=False, opset=args.opset, device="cpu")`.
5. Move the generated file to `--output` only after export succeeds.
6. Run `onnx.checker.check_model`, collect graph metadata and operator summaries, and atomically write `<output>.report.json`.
7. If `--verify-runtime` is enabled, run deterministic CPU parity and return nonzero when shapes differ or `numpy.allclose` fails.
8. Preserve the ONNX file and write a failure report for post-export validation errors.

- [ ] **Step 4: Run export tests and verify GREEN**

Run:

```bash
python tests/test_fswd_deployment_tools.py ExportCliTests -v
```

Expected: all `ExportCliTests` pass without importing framework packages.

- [ ] **Step 5: Run syntax and help smoke checks**

```bash
python -m py_compile scripts/fswd_deploy_common.py scripts/export_fswd_onnx.py
python scripts/export_fswd_onnx.py --help
```

Expected: exit code 0; help lists `--weights`, `--output`, `--imgsz`, `--opset`, `--verify-runtime`, and `--overwrite`.

- [ ] **Step 6: Commit the export CLI**

```bash
git add scripts/export_fswd_onnx.py tests/test_fswd_deployment_tools.py
git commit -m "feat: export and validate FSWD ONNX models"
```

### Task 3: Target-Neutral Vitis Inspector CLI

**Files:**
- Create: `scripts/inspect_fswd_vitis.py`
- Modify: `tests/test_fswd_deployment_tools.py`

- [ ] **Step 1: Write failing tests for required target and manifest protection**

```python
class InspectorCliTests(unittest.TestCase):
    def test_target_is_required(self):
        parser = inspect_cli.build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["--weights", "best.pt", "--output-dir", "inspect"])

    def test_inspector_defaults_are_fixed_cpu_inputs(self):
        parser = inspect_cli.build_parser()
        args = parser.parse_args(["--weights", "best.pt", "--target", "TARGET", "--output-dir", "inspect"])
        self.assertEqual(args.imgsz, 640)
        self.assertEqual(args.verbose_level, 2)
        self.assertEqual(args.image_format, "svg")

    def test_existing_manifest_requires_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "inspection_manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "--overwrite"):
                inspect_cli.prepare_manifest(Path(directory), overwrite=False)
```

- [ ] **Step 2: Run Inspector tests and verify RED**

Run:

```bash
python tests/test_fswd_deployment_tools.py InspectorCliTests -v
```

Expected: import failure because `scripts.inspect_fswd_vitis` does not exist.

- [ ] **Step 3: Implement the Inspector CLI**

Implement these public interfaces with consistent signatures: `build_parser() -> argparse.ArgumentParser`, `inspector_requirements() -> Dict[str, str]`, `prepare_manifest(output_dir: Path, overwrite: bool) -> Path`, `run(args: argparse.Namespace) -> int`, and `main() -> int`.

`run` validates the checkpoint, preflights `torch`, `ultralytics`, and `pytorch_nndct`, sets `YOLO_AUTOINSTALL=false`, loads `YOLO(weights).model.float().eval().cpu()`, creates `torch.randn(1, 3, imgsz, imgsz)`, and calls:

```python
Inspector(args.target).inspect(
    model,
    (dummy,),
    device=torch.device("cpu"),
    output_dir=str(args.output_dir),
    verbose_level=args.verbose_level,
    image_format=None if args.image_format == "none" else args.image_format,
)
```

It writes an atomic success or failure manifest with target, input shape, checkpoint hash, Git/runtime metadata, arguments, timestamp, status, and error details. It never supplies a default target.

- [ ] **Step 4: Run Inspector tests and verify GREEN**

Run:

```bash
python tests/test_fswd_deployment_tools.py InspectorCliTests -v
```

Expected: all `InspectorCliTests` pass without importing NNDCT.

- [ ] **Step 5: Run syntax and help smoke checks**

```bash
python -m py_compile scripts/inspect_fswd_vitis.py
python scripts/inspect_fswd_vitis.py --help
```

Expected: exit code 0; help marks `--target` as required.

- [ ] **Step 6: Commit the Inspector CLI**

```bash
git add scripts/inspect_fswd_vitis.py tests/test_fswd_deployment_tools.py
git commit -m "feat: inspect FSWD models with Vitis AI"
```

### Task 4: Deployment Documentation and Environment-Gated Checks

**Files:**
- Create: `docs/quan_dev/README.md`
- Modify: `tests/test_fswd_deployment_tools.py`

- [ ] **Step 1: Add environment-gated ONNX graph inspection test**

```python
@unittest.skipUnless(importlib.util.find_spec("onnx"), "onnx is not installed")
class OnnxIntegrationTests(unittest.TestCase):
    def test_graph_report_reads_real_onnx_nodes(self):
        import onnx
        from onnx import TensorProto, helper

        graph = helper.make_graph(
            [helper.make_node("Div", ["input", "scale"], ["output"])],
            "test",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1])],
            [helper.make_tensor("scale", TensorProto.FLOAT, [1], [2.0])],
        )
        report = export_cli.onnx_graph_report(helper.make_model(graph), onnx)
        self.assertEqual(report["operators"]["review_operations"], {"Div": 1})
```

- [ ] **Step 2: Run all unit tests**

```bash
python tests/test_fswd_deployment_tools.py -v
```

Expected: dependency-free tests pass; ONNX integration test passes when ONNX exists or reports one explicit skip.

- [ ] **Step 3: Write the operational guide**

Document exact commands for:

```bash
# Training/export host
python scripts/export_fswd_onnx.py --weights /data/best.pt --output /data/fswd-yolo.onnx --imgsz 640 --opset 13
python scripts/export_fswd_onnx.py --weights /data/best.pt --output /data/fswd-yolo.onnx --verify-runtime --overwrite

# Pristine Vitis AI 3.5 container
python -c "import torch, pytorch_nndct; print(torch.__version__)"
python scripts/inspect_fswd_vitis.py --weights /workspace/weights/best.pt --target ACTUAL_TARGET --output-dir /workspace/inspect_fswd
```

The guide explains reports, review-sensitive operators, how to obtain a real target after board selection, why ONNX export does not prove DPU compatibility, and why `conda install timm/onnx` must not be used in the Vitis environment. It includes explicit, separated examples such as `python -m pip install --no-deps einops==0.8.0 timm==0.6.7` for the Vitis container and `python -m pip install --no-deps onnx==1.14.0` only when ONNX is required there, each followed by an immediate NNDCT import regression check. The tools themselves install nothing.

- [ ] **Step 4: Verify documentation references and diff hygiene**

```bash
rg -n "export_fswd_onnx|inspect_fswd_vitis|ACTUAL_TARGET|AUTOINSTALL" docs/quan_dev/README.md scripts tests/test_fswd_deployment_tools.py
git diff --check
```

Expected: all documented commands resolve to repository files and `git diff --check` reports no whitespace errors.

- [ ] **Step 5: Commit documentation and integration coverage**

```bash
git add docs/quan_dev/README.md tests/test_fswd_deployment_tools.py
git commit -m "docs: document FSWD ONNX and Vitis workflow"
```

### Task 5: Final Verification and Remote Delivery

**Files:**
- Verify all files above.

- [ ] **Step 1: Run complete local verification**

```bash
python tests/test_fswd_deployment_tools.py -v
python -m py_compile scripts/fswd_deploy_common.py scripts/export_fswd_onnx.py scripts/inspect_fswd_vitis.py tests/test_fswd_deployment_tools.py
python scripts/export_fswd_onnx.py --help
python scripts/inspect_fswd_vitis.py --help
git diff --check origin/revision...HEAD
git status --short --branch
```

Expected: unit tests pass with only documented environment skips, all modules compile, both help commands exit 0, diff check is clean, and only the user's untracked `docs/quan_dev/logs.txt` remains outside commits.

- [ ] **Step 2: Inspect commit scope**

```bash
git log --oneline origin/revision..HEAD
git diff --stat origin/revision...HEAD
git diff --name-status origin/revision...HEAD
```

Expected: commits contain only the approved design, plan, deployment tools, tests, and guide.

- [ ] **Step 3: Push the feature branch**

```bash
git push -u origin codex/fswd-onnx-deployment-tools
```

Expected: remote branch is created and upstream tracking is configured.

- [ ] **Step 4: Report submodule delivery without updating the paper pointer**

Report branch, commits, push status, test status, environment-gated checks, and that the paper-workspace submodule pointer remains intentionally uncommitted.
