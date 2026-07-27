"""Tests for the DPU-oriented FSWD-YOLO model variant."""

from __future__ import annotations

import ast
import importlib.util
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DPU_MODULE = REPO_ROOT / "ultralytics" / "nn" / "modules" / "fswd_dpu.py"
DPU_YAML = REPO_ROOT / "ultralytics" / "cfg" / "models" / "11" / "fswd-yolo-dpu.yaml"
ORIGINAL_YAML = REPO_ROOT / "ultralytics" / "cfg" / "models" / "11" / "fswd-yolo.yaml"
MODULE_EXPORTS = REPO_ROOT / "ultralytics" / "nn" / "modules" / "__init__.py"
TASKS_MODULE = REPO_ROOT / "ultralytics" / "nn" / "tasks.py"
MIGRATION_MODULE = REPO_ROOT / "scripts" / "fswd_dpu_migration.py"
MODEL_ADAPTER = REPO_ROOT / "scripts" / "fswd_dpu_model.py"
DETECT_HEAD = REPO_ROOT / "ultralytics" / "nn" / "modules" / "head.py"


class DPUModelStaticContractTests(unittest.TestCase):
    """Dependency-light checks that keep the deployment graph contract reviewable."""

    def _module_tree(self) -> ast.Module:
        self.assertTrue(DPU_MODULE.is_file(), f"missing DPU module: {DPU_MODULE}")
        return ast.parse(DPU_MODULE.read_text(encoding="utf-8"))

    def test_dpu_module_defines_approved_classes(self):
        tree = self._module_tree()
        classes = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}

        self.assertTrue(
            {
                "SplitFreeC2f",
                "C3k2DPU",
                "C3k2GhostSimAMinnerDPU",
                "DPUSpatialAttentionBlock",
                "DPUChannelAttentionBlock",
                "C2PSFCADPU",
            }.issubset(classes)
        )

    def test_dpu_forward_paths_do_not_call_split_or_chunk(self):
        tree = self._module_tree()
        forbidden = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr in {"split", "chunk"}:
                forbidden.append((node.func.attr, node.lineno))

        self.assertEqual(forbidden, [])

    def test_c2psfca_keeps_three_independent_projections(self):
        source = DPU_MODULE.read_text(encoding="utf-8") if DPU_MODULE.is_file() else ""

        self.assertIn("self.keep_projection", source)
        self.assertIn("self.spatial_projection", source)
        self.assertIn("self.channel_projection", source)
        self.assertIn("nn.Hardswish", source)
        self.assertIn("nn.Hardsigmoid", source)
        self.assertIn("nn.AdaptiveAvgPool2d", source)

    def test_dpu_classes_are_exported_and_registered(self):
        exports = MODULE_EXPORTS.read_text(encoding="utf-8")
        tasks = TASKS_MODULE.read_text(encoding="utf-8")

        self.assertIn("from .fswd_dpu import", exports)
        for class_name in ("C2PSFCADPU", "C3k2DPU", "C3k2GhostSimAMinnerDPU"):
            self.assertIn(class_name, exports)
            self.assertIn(class_name, tasks)

        self.assertIn("if m in {C3k2, C3k2DPU}", tasks)

    def test_dpu_yaml_preserves_topology_and_selects_hardswish(self):
        self.assertTrue(DPU_YAML.is_file(), f"missing DPU YAML: {DPU_YAML}")
        source = DPU_YAML.read_text(encoding="utf-8")

        self.assertIn("activation: torch.nn.Hardswish()", source)
        self.assertEqual(source.count("C3k2GhostSimAMinnerDPU"), 2)
        self.assertEqual(source.count("C3k2DPU"), 5)
        self.assertEqual(source.count("C2PSFCADPU"), 1)
        self.assertEqual(source.count("Detect, [nc]"), 1)

        original_connections = [
            line.strip().split("#", 1)[0].rstrip()
            for line in ORIGINAL_YAML.read_text(encoding="utf-8").splitlines()
            if line.lstrip().startswith("- [")
        ]
        candidate_connections = [
            line.strip().split("#", 1)[0].rstrip()
            for line in source.splitlines()
            if line.lstrip().startswith("- [")
        ]
        self.assertEqual(len(candidate_connections), len(original_connections))

    def test_migration_module_exposes_audited_conversion_api(self):
        self.assertTrue(MIGRATION_MODULE.is_file(), f"missing migration module: {MIGRATION_MODULE}")
        tree = ast.parse(MIGRATION_MODULE.read_text(encoding="utf-8"))
        functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}

        self.assertTrue(
            {
                "copy_conv_output_slice",
                "copy_matching_state",
                "migrate_split_free_module",
                "migrate_c2psfca",
                "migrate_fswd_model",
            }.issubset(functions)
        )

    def test_detect_head_exposes_explicit_raw_output_contract(self):
        source = DETECT_HEAD.read_text(encoding="utf-8")

        self.assertIn("self.raw_output = False", source)
        self.assertIn('getattr(self, "raw_output", False)', source)

    def test_raw_detect_adapter_exposes_host_decode_api(self):
        self.assertTrue(MODEL_ADAPTER.is_file(), f"missing model adapter: {MODEL_ADAPTER}")
        source = MODEL_ADAPTER.read_text(encoding="utf-8")
        tree = ast.parse(source)
        classes = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
        functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}

        self.assertIn("RawDetectHeadAdapter", classes)
        self.assertTrue(
            {"find_detect_head", "decode_raw_predictions", "compare_decode_parity"}.issubset(functions)
        )
        self.assertIn('getattr(self.detect, "raw_output", False)', source)


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required for DPU model runtime tests")
class DPUModelRuntimeTests(unittest.TestCase):
    """Runtime checks executed on the activated FSWD-YOLO model host."""

    def test_c2psfca_shape_gradients_and_branch_shapes(self):
        import torch

        from ultralytics.nn.modules.fswd_dpu import C2PSFCADPU

        torch.manual_seed(0)
        module = C2PSFCADPU(128, 128, n=1, e=0.5, channel_reduction=4)
        branch_shapes = {}
        hooks = []
        for name in ("keep_projection", "spatial_projection", "channel_projection"):
            projection = getattr(module, name)
            hooks.append(
                projection.register_forward_hook(
                    lambda _module, _inputs, output, branch=name: branch_shapes.setdefault(branch, tuple(output.shape))
                )
            )

        x = torch.randn(2, 128, 20, 20, requires_grad=True)
        y = module(x)
        y.mean().backward()
        for hook in hooks:
            hook.remove()

        self.assertEqual(tuple(y.shape), tuple(x.shape))
        self.assertTrue(torch.isfinite(y).all())
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertEqual(
            branch_shapes,
            {
                "keep_projection": (2, 64, 20, 20),
                "spatial_projection": (2, 64, 20, 20),
                "channel_projection": (2, 64, 20, 20),
            },
        )

    def test_c2psfca_uses_dpu_friendly_attention_primitives(self):
        import torch

        from ultralytics.nn.modules.fswd_dpu import C2PSFCADPU

        module = C2PSFCADPU(128, 128)
        modules = tuple(module.modules())
        self.assertTrue(any(isinstance(item, torch.nn.Hardswish) for item in modules))
        self.assertTrue(any(isinstance(item, torch.nn.Hardsigmoid) for item in modules))
        self.assertTrue(any(isinstance(item, torch.nn.AdaptiveAvgPool2d) for item in modules))
        self.assertTrue(
            any(
                isinstance(item, torch.nn.Conv2d)
                and item.kernel_size == (5, 5)
                and item.groups == item.in_channels
                for item in modules
            )
        )

    def test_dpu_yaml_constructs_without_activation_leakage(self):
        import torch

        from ultralytics import YOLO
        from ultralytics.nn.modules import C2PSFCADPU, C3k2DPU, C3k2GhostSimAMinnerDPU, Conv

        previous = Conv.default_act
        try:
            dpu_model = YOLO(str(DPU_YAML)).model
            modules = tuple(dpu_model.modules())
            self.assertEqual(sum(isinstance(item, C2PSFCADPU) for item in modules), 1)
            self.assertEqual(sum(isinstance(item, C3k2DPU) for item in modules), 5)
            self.assertEqual(sum(isinstance(item, C3k2GhostSimAMinnerDPU) for item in modules), 2)
            self.assertTrue(
                all(
                    isinstance(item.act, (torch.nn.Hardswish, torch.nn.Identity))
                    for item in modules
                    if isinstance(item, Conv)
                )
            )
            self.assertIs(Conv.default_act, previous)

            original_model = YOLO(str(ORIGINAL_YAML)).model
            self.assertTrue(
                any(
                    isinstance(item.act, torch.nn.SiLU)
                    for item in original_model.modules()
                    if isinstance(item, Conv)
                )
            )
            self.assertIs(Conv.default_act, previous)
        finally:
            Conv.default_act = previous

    def test_split_free_projection_migration_is_exact(self):
        import torch

        from scripts.fswd_dpu_migration import migrate_split_free_module
        from ultralytics.nn.modules import C3k2, C3k2GhostSimAMinner
        from ultralytics.nn.modules.fswd_dpu import C3k2DPU, C3k2GhostSimAMinnerDPU

        pairs = (
            (C3k2(64, 64, n=1), C3k2DPU(64, 64, n=1)),
            (
                C3k2GhostSimAMinner(64, 64, n=1),
                C3k2GhostSimAMinnerDPU(64, 64, n=1),
            ),
        )
        x = torch.randn(1, 64, 16, 16)
        for source, destination in pairs:
            source.eval()
            destination.eval()
            summary = migrate_split_free_module(source, destination)
            with torch.no_grad():
                expected = source(x)
                actual = destination(x)
            self.assertTrue(torch.equal(expected, actual))
            self.assertGreater(len(summary.sliced), 0)
            self.assertEqual(summary.unexpected, [])

    def test_c2psfca_migration_accounts_for_redesigned_attention(self):
        from scripts.fswd_dpu_migration import migrate_c2psfca
        from ultralytics.nn.modules import C2PSFCA
        from ultralytics.nn.modules.fswd_dpu import C2PSFCADPU

        summary = migrate_c2psfca(C2PSFCA(128, 128), C2PSFCADPU(128, 128))

        self.assertGreaterEqual(len(summary.sliced), 3)
        self.assertTrue(any(name.startswith("m_psa.") for name in summary.unmapped))
        self.assertTrue(any(name.startswith("m_fca.") for name in summary.unmapped))
        self.assertTrue(any(name.startswith("spatial_blocks.") for name in summary.new))
        self.assertTrue(any(name.startswith("channel_blocks.") for name in summary.new))
        self.assertEqual(summary.unexpected, [])

    def test_raw_detect_adapter_matches_native_decode(self):
        import torch

        from scripts.fswd_dpu_model import RawDetectHeadAdapter, compare_decode_parity, find_detect_head
        from ultralytics import YOLO

        model = YOLO(str(DPU_YAML)).model.float().eval()
        adapter = RawDetectHeadAdapter(model).eval()
        detect = find_detect_head(model)
        input_tensor = torch.randn(1, 3, 640, 640)

        self.assertFalse(detect.raw_output)
        with torch.no_grad():
            raw_outputs = adapter(input_tensor)
        self.assertFalse(detect.raw_output)
        self.assertIsInstance(raw_outputs, tuple)
        self.assertEqual(len(raw_outputs), 3)
        self.assertEqual([tuple(item.shape[-2:]) for item in raw_outputs], [(80, 80), (40, 40), (20, 20)])
        self.assertEqual(
            [id(parameter) for parameter in adapter.parameters()],
            [id(parameter) for parameter in model.parameters()],
        )

        parity = compare_decode_parity(model, input_tensor)
        self.assertTrue(parity["shape_matches"])
        self.assertTrue(parity["allclose"])


if __name__ == "__main__":
    unittest.main()
