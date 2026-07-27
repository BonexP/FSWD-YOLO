# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import fswd_deploy_common as common  # noqa: E402
from scripts import export_fswd_onnx as export_cli  # noqa: E402
from scripts import inspect_fswd_vitis as inspect_cli  # noqa: E402
from scripts import initialize_fswd_dpu as initialize_cli  # noqa: E402
from scripts import profile_fswd_dpu_candidate as profile_cli  # noqa: E402
from scripts import fswd_vitis_report as vitis_report  # noqa: E402


class CommonUtilitiesTests(unittest.TestCase):
    def test_add_repo_root_to_path_exposes_sibling_source_package(self):
        with tempfile.TemporaryDirectory() as directory:
            repo_root = Path(directory)
            script = repo_root / "scripts" / "tool.py"
            package = repo_root / "local_fswd_package"
            script.parent.mkdir()
            script.write_text("", encoding="utf-8")
            package.mkdir()
            (package / "__init__.py").write_text("", encoding="utf-8")

            inserted = common.add_repo_root_to_path(script)
            try:
                self.assertEqual(inserted, repo_root.resolve())
                self.assertIsNotNone(importlib.util.find_spec("local_fswd_package"))
            finally:
                sys.path.remove(str(repo_root.resolve()))

    def test_normalize_version_attribute_makes_torch_version_hashable(self):
        class UnhashableVersion(str):
            __hash__ = None

        module = SimpleNamespace(__version__=UnhashableVersion("1.10.0+cpu"))

        normalized = common.normalize_version_attribute(module)

        self.assertEqual(normalized, "1.10.0+cpu")
        self.assertEqual(type(module.__version__), str)
        self.assertEqual(hash(module.__version__), hash("1.10.0+cpu"))

    def test_validate_weights_requires_existing_pt_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.bin"
            path.write_bytes(b"weights")

            with self.assertRaisesRegex(ValueError, "must use the .pt suffix"):
                common.validate_weights(path)

    def test_validate_weights_requires_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.pt"

            with self.assertRaisesRegex(FileNotFoundError, "does not exist"):
                common.validate_weights(path)

    def test_validate_model_config_requires_existing_yaml_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            invalid = root / "model.txt"
            invalid.write_text("nc: 6", encoding="utf-8")
            valid = root / "model.yaml"
            valid.write_text("nc: 6", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "yaml or .yml"):
                common.validate_model_config(invalid)
            self.assertEqual(common.validate_model_config(valid), valid.resolve())

    def test_prepare_output_rejects_existing_file_without_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "model.onnx"
            output.write_bytes(b"existing")

            with self.assertRaisesRegex(FileExistsError, "--overwrite"):
                common.prepare_output(output, overwrite=False)

    def test_prepare_output_requires_expected_suffix(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "model.bin"

            with self.assertRaisesRegex(ValueError, "must use the .onnx suffix"):
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
            self.assertEqual(list(path.parent.glob(f".{path.name}.*.tmp")), [])

    def test_write_json_atomic_respects_overwrite_protection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text("{}", encoding="utf-8")

            with self.assertRaisesRegex(FileExistsError, "--overwrite"):
                common.write_json_atomic(path, {"status": "ok"}, overwrite=False)

    def test_summarize_operators_counts_review_operations(self):
        summary = common.summarize_operators(["Conv", "Div", "Conv", "MatMul", "Transpose"])

        self.assertEqual(summary["histogram"], {"Conv": 2, "Div": 1, "MatMul": 1, "Transpose": 1})
        self.assertEqual(summary["review_operations"], {"Div": 1, "MatMul": 1, "Transpose": 1})

    def test_require_modules_reports_missing_dependency_without_installing(self):
        missing_name = "fswd_package_that_does_not_exist"

        with self.assertRaises(common.DependencyError) as context:
            common.require_modules({missing_name: "manual installation only"}, "unit test")

        message = str(context.exception)
        self.assertIn(missing_name, message)
        self.assertIn("manual installation only", message)
        self.assertIn(sys.executable, message)
        self.assertIn("does not install packages", message)


class ExportCliTests(unittest.TestCase):
    def test_direct_script_discovers_repository_ultralytics_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / "best.pt"
            output = root / "model.onnx"
            weights.write_bytes(b"checkpoint")

            result = subprocess.run(
                [
                    sys.executable,
                    "-S",
                    str(REPO_ROOT / "scripts" / "export_fswd_onnx.py"),
                    "--weights",
                    str(weights),
                    "--output",
                    str(output),
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("  - torch:", result.stderr)
            self.assertIn("  - onnx:", result.stderr)
            self.assertNotIn("  - ultralytics:", result.stderr)

    def test_export_defaults_are_fpga_review_friendly(self):
        parser = export_cli.build_parser()

        args = parser.parse_args(["--weights", "best.pt", "--output", "fswd.onnx"])

        self.assertEqual(args.imgsz, 640)
        self.assertEqual(args.opset, 13)
        self.assertEqual(args.seed, 0)
        self.assertEqual(args.rtol, 1e-4)
        self.assertEqual(args.atol, 1e-5)
        self.assertFalse(args.verify_runtime)
        self.assertFalse(args.overwrite)

    def test_export_requirements_add_onnxruntime_only_for_parity(self):
        base = export_cli.export_requirements(False)
        parity = export_cli.export_requirements(True)

        self.assertEqual(set(base), {"torch", "onnx", "ultralytics"})
        self.assertNotIn("onnxruntime", base)
        self.assertIn("onnxruntime", parity)

    def test_report_path_appends_report_json(self):
        self.assertEqual(export_cli.report_path(Path("model.onnx")), Path("model.onnx.report.json"))

    def test_onnx_version_must_match_ultralytics_export_range(self):
        export_cli.validate_onnx_version("1.14.0")
        with self.assertRaisesRegex(common.DependencyError, "onnx>=1.12.0,<1.18.0"):
            export_cli.validate_onnx_version("1.18.0")

    def test_export_kwargs_disable_dynamic_nms_and_simplification(self):
        kwargs = export_cli.export_kwargs(imgsz=640, opset=13)

        self.assertEqual(
            kwargs,
            {
                "format": "onnx",
                "imgsz": 640,
                "batch": 1,
                "dynamic": False,
                "nms": False,
                "simplify": False,
                "half": False,
                "opset": 13,
                "device": "cpu",
            },
        )

    def test_normalize_torch_prediction_uses_first_tensor_like_value(self):
        class TensorLike:
            shape = (1, 10, 8400)

            def detach(self):
                return self

        tensor = TensorLike()

        self.assertIs(export_cli.normalize_torch_prediction((tensor, ["features"])), tensor)

    def test_normalize_torch_prediction_rejects_unknown_structure(self):
        with self.assertRaisesRegex(TypeError, "prediction tensor"):
            export_cli.normalize_torch_prediction({"unexpected": "output"})

    def test_dependency_failure_writes_failure_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / "best.pt"
            output = root / "model.onnx"
            weights.write_bytes(b"checkpoint")
            args = export_cli.build_parser().parse_args(
                ["--weights", str(weights), "--output", str(output)]
            )

            with mock.patch.object(
                export_cli.common,
                "require_modules",
                side_effect=common.DependencyError("missing test dependency"),
            ):
                with self.assertRaisesRegex(common.DependencyError, "missing test dependency"):
                    export_cli.run(args)

            payload = json.loads(export_cli.report_path(output).read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "failed")
            self.assertEqual(payload["error"]["type"], "DependencyError")
            self.assertFalse(output.exists())


class InspectorCliTests(unittest.TestCase):
    def test_prediction_comparison_records_shape_and_error_metrics(self):
        class FakeScalar:
            def __init__(self, value):
                self.value = value

            def item(self):
                return self.value

        class FakeDifference:
            def abs(self):
                return self

            def max(self):
                return FakeScalar(0.0002)

            def mean(self):
                return FakeScalar(0.00001)

        class FakeTensor:
            shape = (1, 10, 8400)

            def __sub__(self, _other):
                return FakeDifference()

        fake_torch = SimpleNamespace(allclose=lambda *_args, **_kwargs: True)

        report = inspect_cli.compare_prediction_tensors(
            fake_torch, FakeTensor(), FakeTensor(), rtol=1e-5, atol=1e-6
        )

        self.assertEqual(report["before_shape"], [1, 10, 8400])
        self.assertEqual(report["after_shape"], [1, 10, 8400])
        self.assertTrue(report["shape_matches"])
        self.assertTrue(report["allclose"])
        self.assertEqual(report["max_absolute_error"], 0.0002)
        self.assertEqual(report["mean_absolute_error"], 0.00001)
        self.assertEqual(report["rtol"], 1e-5)
        self.assertEqual(report["atol"], 1e-6)

    def test_model_preparation_disables_inplace_activations_without_changing_c2f(self):
        class FakeSiLU:
            def __init__(self, inplace):
                self.inplace = inplace

        class FakeHardswish:
            def __init__(self, inplace):
                self.inplace = inplace

        class FakeC2f:
            def forward(self, value):
                return ("chunk", value)

            def forward_split(self, value):
                return ("split", value)

        class Unrelated:
            pass

        inplace_silu = FakeSiLU(inplace=True)
        safe_silu = FakeSiLU(inplace=False)
        inplace_hardswish = FakeHardswish(inplace=True)
        c2f = FakeC2f()
        unrelated = Unrelated()
        model = SimpleNamespace(
            modules=lambda: [model, inplace_silu, safe_silu, inplace_hardswish, c2f, unrelated]
        )

        changes = inspect_cli.prepare_model_for_vitis_inspection(model, FakeSiLU, FakeHardswish)

        self.assertEqual(
            changes,
            {
                "silu_inplace_disabled": 1,
                "hardswish_inplace_disabled": 1,
                "c2f_forward_split_enabled": 0,
            },
        )
        self.assertFalse(inplace_silu.inplace)
        self.assertFalse(safe_silu.inplace)
        self.assertFalse(inplace_hardswish.inplace)
        self.assertEqual(c2f.forward("input"), ("chunk", "input"))
        self.assertFalse(hasattr(unrelated, "inplace"))

    def test_activation_experiment_rebinds_unique_silu_modules(self):
        class FakeSiLU:
            def forward(self, value):
                return ("silu", value)

        class Unrelated:
            def forward(self, value):
                return ("unrelated", value)

        def hard_forward(_module, value):
            return ("hardswish", value)

        first = FakeSiLU()
        second = FakeSiLU()
        unrelated = Unrelated()
        model = SimpleNamespace(modules=lambda: [model, first, second, unrelated])

        changes = inspect_cli.apply_activation_experiment(
            model, FakeSiLU, "hardswish", hard_forward
        )

        self.assertEqual(changes, {"silu_activation_replaced": 2})
        self.assertEqual(first.forward("input"), ("hardswish", "input"))
        self.assertEqual(second.forward("input"), ("hardswish", "input"))
        self.assertEqual(unrelated.forward("input"), ("unrelated", "input"))

        copied = copy.deepcopy(first)
        self.assertIs(copied.forward.__self__, copied)
        self.assertEqual(copied.forward("input"), ("hardswish", "input"))

    def test_activation_experiment_rejects_none(self):
        model = SimpleNamespace(modules=lambda: [])

        with self.assertRaisesRegex(ValueError, "explicit hard activation"):
            inspect_cli.apply_activation_experiment(model, object, "none", lambda *_args: None)

    def test_activation_forward_factory_selects_requested_functional(self):
        functional = SimpleNamespace(
            hardswish=lambda value: ("hardswish", value),
            hardsigmoid=lambda value: ("hardsigmoid", value),
        )

        hardswish = inspect_cli.make_activation_forward(functional, "hardswish")
        hardsigmoid = inspect_cli.make_activation_forward(functional, "hardsigmoid")

        self.assertEqual(hardswish(None, "input"), ("hardswish", "input"))
        self.assertEqual(hardsigmoid(None, "input"), ("hardsigmoid", "input"))

    def test_activation_mode_metadata_distinguishes_audit_and_experiment(self):
        audit = inspect_cli.activation_mode_metadata("none")
        experiment = inspect_cli.activation_mode_metadata("hardsigmoid")

        self.assertEqual(audit["mode"], "original_model_audit")
        self.assertEqual(audit["activation_experiment"], "none")
        self.assertTrue(audit["semantic_equivalence_required"])
        self.assertFalse(audit["checkpoint_modified"])
        self.assertEqual(experiment["mode"], "deployment_activation_experiment")
        self.assertEqual(experiment["activation_experiment"], "hardsigmoid")
        self.assertFalse(experiment["semantic_equivalence_required"])
        self.assertFalse(experiment["checkpoint_modified"])

    def test_prediction_gate_always_rejects_shape_mismatch(self):
        comparison = {"shape_matches": False, "allclose": False}

        with self.assertRaisesRegex(inspect_cli.ModelPreparationError, "shape"):
            inspect_cli.enforce_prediction_comparison(
                comparison, semantic_equivalence_required=False
            )

    def test_prediction_gate_requires_allclose_only_for_original_audit(self):
        comparison = {"shape_matches": True, "allclose": False}

        with self.assertRaisesRegex(inspect_cli.ModelPreparationError, "changed raw predictions"):
            inspect_cli.enforce_prediction_comparison(
                comparison, semantic_equivalence_required=True
            )

        inspect_cli.enforce_prediction_comparison(
            comparison, semantic_equivalence_required=False
        )

    def test_model_preparation_payload_marks_non_equivalent_experiment(self):
        comparison = {"shape_matches": True, "allclose": False}

        payload = inspect_cli.build_model_preparation_payload(
            "hardswish",
            {"silu_inplace_disabled": 1, "silu_activation_replaced": 1},
            comparison,
        )

        self.assertEqual(payload["mode"], "deployment_activation_experiment")
        self.assertFalse(payload["semantic_equivalence_required"])
        self.assertFalse(payload["checkpoint_modified"])
        self.assertEqual(payload["changes"]["silu_activation_replaced"], 1)
        self.assertIs(payload["prediction_comparison"], comparison)

    def test_vitis_35_compatibility_reports_unknown_3d_permute(self):
        class FakeOp:
            type = "permute"

            class AttrName:
                ORDER = "order"

        class FakeNode:
            name = "model::swim_transpose_0"
            op = FakeOp()

            @staticmethod
            def node_attr(_name):
                return [0, 2, 1]

        class BuggyInspectorImpl:
            def __init__(self):
                self._node_msgs = defaultdict(set)

            def _attach_extra_node_msg(self, graph):
                transpose_order_to_msg = {}
                node = graph.nodes[0]
                order = node.node_attr(node.op.AttrName.ORDER)
                self._node_msgs[node].add(transpose_order_to_msg[tuple(order)])

        node = FakeNode()
        graph = SimpleNamespace(nodes=[node])

        applied = inspect_cli.install_vitis_35_permute_report_compatibility(
            BuggyInspectorImpl, "permute"
        )
        inspector = BuggyInspectorImpl()
        inspector._attach_extra_node_msg(graph)

        self.assertTrue(applied)
        self.assertIn("permutation order (0, 2, 1)", next(iter(inspector._node_msgs[node])))

    def test_vitis_35_compatibility_leaves_fixed_implementation_unchanged(self):
        class FixedInspectorImpl:
            def _attach_extra_node_msg(self, _graph):
                transpose_order_to_msg = {}
                return transpose_order_to_msg.get((0, 2, 1), "already safe")

        original = FixedInspectorImpl._attach_extra_node_msg

        applied = inspect_cli.install_vitis_35_permute_report_compatibility(
            FixedInspectorImpl, "permute"
        )

        self.assertFalse(applied)
        self.assertIs(FixedInspectorImpl._attach_extra_node_msg, original)

    def test_target_is_required(self):
        parser = inspect_cli.build_parser()

        with self.assertRaises(SystemExit):
            parser.parse_args(["--weights", "best.pt", "--output-dir", "inspect"])

    def test_inspector_defaults_are_fixed_cpu_inputs(self):
        parser = inspect_cli.build_parser()

        args = parser.parse_args(
            ["--weights", "best.pt", "--target", "TARGET", "--output-dir", "inspect"]
        )

        self.assertEqual(args.imgsz, 640)
        self.assertEqual(args.verbose_level, 2)
        self.assertEqual(args.image_format, "svg")
        self.assertEqual(args.activation_experiment, "none")
        self.assertFalse(args.raw_detect_output)
        self.assertFalse(args.overwrite)

    def test_inspector_requires_exactly_one_model_source(self):
        parser = inspect_cli.build_parser()
        required = ["--target", "TARGET", "--output-dir", "inspect"]

        with self.assertRaises(SystemExit):
            parser.parse_args(required)
        with self.assertRaises(SystemExit):
            parser.parse_args(
                ["--weights", "best.pt", "--model-config", "model.yaml", *required]
            )

        config_args = parser.parse_args(["--model-config", "model.yaml", *required])
        self.assertEqual(config_args.model_config, Path("model.yaml"))

    def test_yaml_candidate_rejects_activation_experiments(self):
        parser = inspect_cli.build_parser()
        args = parser.parse_args(
            [
                "--model-config",
                "model.yaml",
                "--target",
                "TARGET",
                "--output-dir",
                "inspect",
                "--activation-experiment",
                "hardswish",
            ]
        )

        with self.assertRaisesRegex(ValueError, "YAML candidate"):
            inspect_cli.validate_source_arguments(args)

    def test_model_source_manifest_distinguishes_weights_and_yaml(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / "best.pt"
            config = root / "model.yaml"
            weights.write_bytes(b"checkpoint")
            config.write_text("nc: 6", encoding="utf-8")

            trained = inspect_cli.model_source_metadata("weights", weights.resolve())
            candidate = inspect_cli.model_source_metadata("model_config", config.resolve())

        self.assertEqual(trained["kind"], "trained_checkpoint")
        self.assertEqual(trained["weight_state"], "trained")
        self.assertEqual(candidate["kind"], "model_config_candidate")
        self.assertEqual(candidate["weight_state"], "random_initialization")
        self.assertEqual(trained["sha256"], hashlib.sha256(b"checkpoint").hexdigest())

    def test_inspector_accepts_raw_detect_output_mode(self):
        parser = inspect_cli.build_parser()
        args = parser.parse_args(
            [
                "--model-config",
                "model.yaml",
                "--target",
                "TARGET",
                "--output-dir",
                "inspect",
                "--raw-detect-output",
            ]
        )

        self.assertTrue(args.raw_detect_output)

    def test_inspector_accepts_only_named_activation_experiments(self):
        parser = inspect_cli.build_parser()
        base = ["--weights", "best.pt", "--target", "TARGET", "--output-dir", "inspect"]

        self.assertEqual(
            parser.parse_args([*base, "--activation-experiment", "hardswish"]).activation_experiment,
            "hardswish",
        )
        self.assertEqual(
            parser.parse_args(
                [*base, "--activation-experiment", "hardsigmoid"]
            ).activation_experiment,
            "hardsigmoid",
        )
        with self.assertRaises(SystemExit):
            parser.parse_args([*base, "--activation-experiment", "silu"])

    def test_existing_manifest_requires_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "inspection_manifest.json"
            manifest.write_text("{}", encoding="utf-8")

            with self.assertRaisesRegex(FileExistsError, "--overwrite"):
                inspect_cli.prepare_manifest(Path(directory), overwrite=False)


class InitializeDPUCliTests(unittest.TestCase):
    def test_migration_semantics_distinguish_mapping_from_model_parity(self):
        semantics = initialize_cli.migration_semantics()

        self.assertFalse(semantics["source_to_candidate_fp32_equivalent"])
        self.assertEqual(
            semantics["gsconv_shuffle_rewrite"],
            "exact_when_activation_modules_match",
        )

    def test_initialize_cli_requires_checkpoint_config_and_output(self):
        parser = initialize_cli.build_parser()

        with self.assertRaises(SystemExit):
            parser.parse_args([])
        args = parser.parse_args(
            [
                "--source-checkpoint",
                "best.pt",
                "--model-config",
                "fswd-yolo-dpu.yaml",
                "--output",
                "fswd-yolo-dpu-init.pt",
            ]
        )
        self.assertEqual(args.imgsz, 640)
        self.assertFalse(args.overwrite)

    def test_initialize_report_path_appends_migration_json(self):
        self.assertEqual(
            initialize_cli.migration_report_path(Path("model.pt")),
            Path("model.pt.migration.json"),
        )

    def test_initialize_artifacts_require_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "model.pt"
            report = Path(directory) / "report.json"
            output.write_bytes(b"existing")

            with self.assertRaisesRegex(FileExistsError, "--overwrite"):
                initialize_cli.prepare_artifacts(output, report, overwrite=False)

    def test_initialize_rejects_non_positive_image_size(self):
        parser = initialize_cli.build_parser()
        args = parser.parse_args(
            [
                "--source-checkpoint",
                "best.pt",
                "--model-config",
                "model.yaml",
                "--output",
                "output.pt",
                "--imgsz",
                "0",
            ]
        )

        with self.assertRaisesRegex(ValueError, "greater than zero"):
            initialize_cli.validate_arguments(args)

    def test_initialize_direct_script_checks_dependencies_without_installing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "best.pt"
            config = root / "model.yaml"
            output = root / "output.pt"
            checkpoint.write_bytes(b"checkpoint")
            config.write_text("nc: 6", encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    "-S",
                    str(REPO_ROOT / "scripts" / "initialize_fswd_dpu.py"),
                    "--source-checkpoint",
                    str(checkpoint),
                    "--model-config",
                    str(config),
                    "--output",
                    str(output),
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )

        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("  - torch:", result.stderr)
        self.assertIn("does not install packages", result.stderr)

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "PyTorch is required")
    def test_initialize_checkpoint_round_trip(self):
        import torch

        from ultralytics import YOLO
        from ultralytics.nn.modules.fswd_dpu import C2PSFCADPU

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.pt"
            output_path = root / "dpu-init.pt"
            report_path = root / "migration.json"
            source_model = YOLO(str(REPO_ROOT / "ultralytics/cfg/models/11/fswd-yolo.yaml")).model
            torch.save({"model": source_model, "train_args": {}}, source_path)
            before_hash = common.sha256_file(source_path)
            args = initialize_cli.build_parser().parse_args(
                [
                    "--source-checkpoint",
                    str(source_path),
                    "--model-config",
                    str(REPO_ROOT / "ultralytics/cfg/models/11/fswd-yolo-dpu.yaml"),
                    "--output",
                    str(output_path),
                    "--report",
                    str(report_path),
                ]
            )

            self.assertEqual(initialize_cli.run(args), 0)
            self.assertEqual(common.sha256_file(source_path), before_hash)
            reloaded = YOLO(str(output_path)).model
            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertTrue(any(isinstance(module, C2PSFCADPU) for module in reloaded.modules()))
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["migration"]["counts"]["unexpected"], 0)
        self.assertFalse(report["migration_semantics"]["source_to_candidate_fp32_equivalent"])
        self.assertTrue(report["decode_parity"]["allclose"])


class ProfileDPUCandidateTests(unittest.TestCase):
    def test_resource_gates_enforce_parameter_and_flop_limits(self):
        passing = profile_cli.evaluate_resource_gates(
            {"parameters": 100, "gflops": 20.0},
            {"parameters": 110, "gflops": 20.0},
        )
        parameter_failure = profile_cli.evaluate_resource_gates(
            {"parameters": 100, "gflops": 20.0},
            {"parameters": 111, "gflops": 19.0},
        )
        flop_failure = profile_cli.evaluate_resource_gates(
            {"parameters": 100, "gflops": 20.0},
            {"parameters": 100, "gflops": 20.1},
        )

        self.assertTrue(passing["parameter_gate"])
        self.assertTrue(passing["flop_gate"])
        self.assertFalse(parameter_failure["parameter_gate"])
        self.assertFalse(flop_failure["flop_gate"])

    def test_resource_gate_counts_persistent_deployment_buffers(self):
        result = profile_cli.evaluate_resource_gates(
            {"parameters": 100, "deployment_tensors": 100, "gflops": 20.0},
            {"parameters": 100, "deployment_tensors": 111, "gflops": 19.0},
        )

        self.assertAlmostEqual(result["parameter_ratio"], 1.11)
        self.assertFalse(result["parameter_gate"])

    def test_profile_cli_defaults_to_640(self):
        args = profile_cli.build_parser().parse_args(
            [
                "--baseline-config",
                "baseline.yaml",
                "--candidate-config",
                "candidate.yaml",
                "--output",
                "profile.json",
            ]
        )

        self.assertEqual(args.imgsz, 640)
        self.assertFalse(args.overwrite)

    def test_profile_direct_script_only_checks_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline.yaml"
            candidate = root / "candidate.yaml"
            output = root / "profile.json"
            baseline.write_text("nc: 6", encoding="utf-8")
            candidate.write_text("nc: 6", encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    "-S",
                    str(REPO_ROOT / "scripts" / "profile_fswd_dpu_candidate.py"),
                    "--baseline-config",
                    str(baseline),
                    "--candidate-config",
                    str(candidate),
                    "--output",
                    str(output),
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )

        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("does not install packages", result.stderr)


class VitisReportTests(unittest.TestCase):
    REPORT_HEADER = """hardware constraints
node name    op Type    hardware constraints
"""
    REPORT_ROWS = """DetectionModel::conv/act    aten::silu    aten::silu can't be converted to XIR.
DetectionModel::conv/transpose    nndct_permute    xir::Op has been assigned to CPU.
DetectionModel::custom    nndct_custom    Target-specific constraint.
"""

    def test_report_summary_deduplicates_and_classifies_repeated_rows(self):
        text = self.REPORT_HEADER + self.REPORT_ROWS * 3

        summary = vitis_report.summarize_inspector_report_text(text)

        self.assertEqual(summary["parsed_row_count"], 9)
        self.assertEqual(summary["unique_row_count"], 3)
        self.assertEqual(summary["repeated_row_count"], 6)
        self.assertEqual(summary["unique_node_count"], 3)
        self.assertEqual(
            summary["category_counts"],
            {"direct_cpu_assignment": 1, "direct_unsupported": 1, "other_cpu_constraint": 1},
        )
        self.assertEqual(
            summary["operators_by_category"],
            {
                "direct_cpu_assignment": {"nndct_permute": 1},
                "direct_unsupported": {"aten::silu": 1},
                "other_cpu_constraint": {"nndct_custom": 1},
            },
        )
        self.assertEqual(summary["direct_blockers"]["nndct_permute"]["count"], 1)
        self.assertEqual(summary["direct_blockers"]["aten::silu"]["count"], 1)
        self.assertEqual(len(summary["direct_blockers"]["aten::silu"]["examples"]), 1)
        self.assertTrue(summary["requires_cpu_fallback"])

    def test_empty_hardware_constraints_table_has_zero_findings(self):
        summary = vitis_report.summarize_inspector_report_text(self.REPORT_HEADER)

        self.assertEqual(summary["parsed_row_count"], 0)
        self.assertEqual(summary["unique_row_count"], 0)
        self.assertFalse(summary["requires_cpu_fallback"])

    def test_target_capability_constraints_are_direct_blockers(self):
        reasons = (
            "xir::Op has been assigned to CPU: "
            "[DPUCZDX8G_ISA1_B4096 does not support eltwise DIV].",
            "xir::Op has been assigned to CPU: "
            '[DPU only supports positive "input_channel"(0)].',
        )

        for reason in reasons:
            with self.subTest(reason=reason):
                self.assertEqual(
                    vitis_report.classify_constraint_reason(reason),
                    "direct_unsupported",
                )

    def test_plain_cpu_assignment_is_a_direct_partition_blocker(self):
        reason = "xir::Op{name = shuffle, type = transpose} has been assigned to CPU."

        self.assertEqual(
            vitis_report.classify_constraint_reason(reason),
            "direct_cpu_assignment",
        )

    def test_file_summary_records_resolved_report_path(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "inspect_TARGET.txt"
            report.write_text(self.REPORT_HEADER + self.REPORT_ROWS, encoding="utf-8")

            summary = vitis_report.summarize_inspector_report(report)

            self.assertEqual(summary["report_path"], str(report.resolve()))
            self.assertEqual(summary["unique_row_count"], 3)

    def test_non_report_text_is_rejected(self):
        with self.assertRaisesRegex(vitis_report.InspectorReportError, "hardware constraints"):
            vitis_report.summarize_inspector_report_text("unrelated non-empty output")

    def test_find_generated_report_detects_new_report(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            before = vitis_report.snapshot_report_signatures(output_dir)
            report = output_dir / "inspect_TARGET.txt"
            report.write_text(self.REPORT_HEADER, encoding="utf-8")

            generated = vitis_report.find_generated_report(output_dir, before)

            self.assertEqual(generated, report.resolve())

    def test_find_generated_report_detects_updated_report(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            report = output_dir / "inspect_TARGET.txt"
            report.write_text("old", encoding="utf-8")
            before = vitis_report.snapshot_report_signatures(output_dir)
            report.write_text(self.REPORT_HEADER, encoding="utf-8")

            generated = vitis_report.find_generated_report(output_dir, before)

            self.assertEqual(generated, report.resolve())

    def test_find_generated_report_rejects_no_changed_report(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            before = vitis_report.snapshot_report_signatures(output_dir)

            with self.assertRaisesRegex(vitis_report.InspectorReportError, "did not generate"):
                vitis_report.find_generated_report(output_dir, before)

    def test_find_generated_report_rejects_multiple_changed_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            before = vitis_report.snapshot_report_signatures(output_dir)
            (output_dir / "inspect_A.txt").write_text(self.REPORT_HEADER, encoding="utf-8")
            (output_dir / "inspect_B.txt").write_text(self.REPORT_HEADER, encoding="utf-8")

            with self.assertRaisesRegex(vitis_report.InspectorReportError, "multiple"):
                vitis_report.find_generated_report(output_dir, before)


@unittest.skipUnless(importlib.util.find_spec("onnx"), "onnx is not installed")
class OnnxIntegrationTests(unittest.TestCase):
    def test_graph_report_reads_real_onnx_nodes(self):
        import onnx
        from onnx import TensorProto, helper

        graph = helper.make_graph(
            [helper.make_node("Div", ["input", "scale"], ["output"])],
            "fswd_test_graph",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1])],
            [helper.make_tensor("scale", TensorProto.FLOAT, [1], [2.0])],
        )

        report = export_cli.onnx_graph_report(helper.make_model(graph), onnx)

        self.assertEqual(report["inputs"][0]["shape"], [1, 1])
        self.assertEqual(report["operators"]["review_operations"], {"Div": 1})


if __name__ == "__main__":
    unittest.main()
