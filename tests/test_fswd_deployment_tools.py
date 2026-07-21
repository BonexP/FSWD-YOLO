# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

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
        self.assertFalse(args.overwrite)

    def test_existing_manifest_requires_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "inspection_manifest.json"
            manifest.write_text("{}", encoding="utf-8")

            with self.assertRaisesRegex(FileExistsError, "--overwrite"):
                inspect_cli.prepare_manifest(Path(directory), overwrite=False)


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
