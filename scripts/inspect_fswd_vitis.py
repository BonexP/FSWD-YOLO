#!/usr/bin/env python3
"""Run Vitis AI Inspector on an FSWD-YOLO PyTorch checkpoint."""

from __future__ import annotations

import argparse
import inspect
import os
import platform
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

try:
    from scripts import fswd_deploy_common as common
except ImportError:  # Direct execution: python scripts/inspect_fswd_vitis.py
    import fswd_deploy_common as common


REPO_ROOT = common.add_repo_root_to_path(Path(__file__))
MODEL_PREPARATION_RTOL = 1e-5
MODEL_PREPARATION_ATOL = 1e-6


class ModelPreparationError(RuntimeError):
    """Raised when a Vitis graph preparation changes model predictions."""


def prepare_model_for_vitis_inspection(
    model: Any, silu_class: Any, c2f_class: Any
) -> Dict[str, int]:
    """Apply semantics-preserving graph forms preferred by Vitis AI Inspector."""
    changes = {"silu_inplace_disabled": 0, "c2f_forward_split_enabled": 0}
    for module in model.modules():
        if isinstance(module, silu_class) and getattr(module, "inplace", False):
            module.inplace = False
            changes["silu_inplace_disabled"] += 1

        if isinstance(module, c2f_class) and callable(getattr(module, "forward_split", None)):
            current_function = getattr(module.forward, "__func__", module.forward)
            split_function = getattr(module.forward_split, "__func__", module.forward_split)
            if current_function is not split_function:
                module.forward = module.forward_split
                changes["c2f_forward_split_enabled"] += 1
    return changes


def compare_prediction_tensors(
    torch_module: Any,
    before: Any,
    after: Any,
    rtol: float,
    atol: float,
) -> Dict[str, Any]:
    """Compare raw predictions before and after Vitis graph preparation."""
    before_shape = list(before.shape)
    after_shape = list(after.shape)
    shape_matches = before_shape == after_shape
    if not shape_matches:
        return {
            "before_shape": before_shape,
            "after_shape": after_shape,
            "shape_matches": False,
            "allclose": False,
            "max_absolute_error": None,
            "mean_absolute_error": None,
            "rtol": rtol,
            "atol": atol,
        }

    absolute_error = (before - after).abs()
    return {
        "before_shape": before_shape,
        "after_shape": after_shape,
        "shape_matches": True,
        "allclose": bool(torch_module.allclose(before, after, rtol=rtol, atol=atol)),
        "max_absolute_error": float(absolute_error.max().item()),
        "mean_absolute_error": float(absolute_error.mean().item()),
        "rtol": rtol,
        "atol": atol,
    }


def install_vitis_35_permute_report_compatibility(
    inspector_impl_class: Any, permute_op: Any
) -> bool:
    """Prevent Vitis AI 3.5 Inspector reports from crashing on 3D permutes."""
    original = inspector_impl_class._attach_extra_node_msg
    try:
        source = inspect.getsource(original)
    except (OSError, TypeError):
        return False
    if "transpose_order_to_msg[tuple(order)]" not in source:
        return False

    transpose_order_to_msg = {
        (0, 3, 1, 2): "from 'NHWC' to 'NCHW'",
        (0, 2, 3, 1): "from 'NCHW' to 'NHWC'",
        (0, 4, 3, 1, 2): "from 'NHWDC' to 'NCDHW'",
        (0, 3, 4, 2, 1): "from 'NCDHW' to 'NHWDC'",
    }

    def attach_extra_node_msg_compat(instance: Any, graph: Any) -> None:
        for node in graph.nodes:
            is_inserted_permute = node.op.type == permute_op and any(
                marker in node.name for marker in ("swim_transpose", "sink_transpose")
            )
            if not is_inserted_permute:
                continue
            order = tuple(node.node_attr(node.op.AttrName.ORDER))
            layout_message = transpose_order_to_msg.get(order)
            if layout_message is None:
                conversion = f"using permutation order {order}"
            else:
                conversion = f"to convert data layout {layout_message}"
            instance._node_msgs[node].add(
                f"quantizer insert this permute operation {conversion} for deployment."
            )

    inspector_impl_class._attach_extra_node_msg = attach_extra_node_msg_compat
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect an FSWD-YOLO PyTorch graph for an explicit Vitis AI target."
    )
    parser.add_argument("--weights", required=True, type=Path, help="Input Ultralytics .pt checkpoint")
    parser.add_argument(
        "--target",
        required=True,
        help="Vitis AI Inspector target name or fingerprint for the intended hardware",
    )
    parser.add_argument("--output-dir", required=True, type=Path, help="Inspector output directory")
    parser.add_argument("--imgsz", type=int, default=640, help="Square input size (default: 640)")
    parser.add_argument(
        "--verbose-level",
        type=int,
        choices=(0, 1, 2),
        default=2,
        help="Inspector verbosity (default: 2)",
    )
    parser.add_argument(
        "--image-format",
        choices=("svg", "png", "none"),
        default="svg",
        help="Inspector diagram format (default: svg)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Inspector input seed (default: 0)")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing manifest")
    return parser


def inspector_requirements() -> Dict[str, str]:
    return {
        "torch": "activate the PyTorch environment supplied for Vitis AI",
        "ultralytics": "install this repository in the active Vitis AI environment",
        "pytorch_nndct": "activate a Vitis AI PyTorch/NNDCT environment containing Inspector",
    }


def prepare_manifest(output_dir: Path, overwrite: bool) -> Path:
    directory = Path(output_dir).expanduser().resolve()
    if directory.exists() and not directory.is_dir():
        raise ValueError(f"Inspector output path is not a directory: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / "inspection_manifest.json"
    if manifest.exists() and not overwrite:
        raise FileExistsError(
            f"Inspector manifest already exists; pass --overwrite to replace it: {manifest}"
        )
    if manifest.exists() and not manifest.is_file():
        raise ValueError(f"Inspector manifest path is not a file: {manifest}")
    return manifest


def _serialize_arguments(args: argparse.Namespace) -> Dict[str, Any]:
    return {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()}


def _base_manifest(args: argparse.Namespace, weights: Path, output_dir: Path) -> Dict[str, Any]:
    return {
        "tool": "inspect_fswd_vitis",
        "status": "running",
        "timestamp_utc": common.utc_now(),
        "arguments": _serialize_arguments(args),
        "target": args.target,
        "input": {
            "weights": str(weights),
            "weights_sha256": common.sha256_file(weights),
            "tensor_shape": [1, 3, args.imgsz, args.imgsz],
            "dtype": "float32",
            "device": "cpu",
        },
        "output_dir": str(output_dir),
        "git": common.git_metadata(REPO_ROOT),
        "runtime": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
        },
        "packages": common.package_versions(
            ["torch", "ultralytics", "pytorch-nndct", "vai-q-pytorch"]
        ),
    }


def run(args: argparse.Namespace) -> int:
    if args.imgsz <= 0:
        raise ValueError("--imgsz must be greater than zero")
    if not args.target.strip():
        raise ValueError("--target must not be empty")

    weights = common.validate_weights(args.weights)
    output_dir = Path(args.output_dir).expanduser().resolve()
    manifest = prepare_manifest(output_dir, args.overwrite)
    payload = _base_manifest(args, weights, output_dir)

    os.environ["YOLO_AUTOINSTALL"] = "false"
    try:
        common.require_modules(inspector_requirements(), "FSWD-YOLO Vitis AI inspection")

        import torch

        common.normalize_version_attribute(torch)
        from nndct_shared.base import NNDCT_OP
        from pytorch_nndct.apis import Inspector
        from pytorch_nndct.hardware_v3.inspector import InspectorImpl
        from ultralytics import YOLO
        from ultralytics.nn.modules import C2f
        from scripts.export_fswd_onnx import normalize_torch_prediction

        compatibility_applied = install_vitis_35_permute_report_compatibility(
            InspectorImpl, NNDCT_OP.PERMUTE
        )
        payload["compatibility"] = {
            "vitis_35_permute_report_patch_applied": compatibility_applied,
            "scope": "Inspector report messages only",
        }

        model = YOLO(str(weights)).model.float().eval().cpu()
        torch.manual_seed(args.seed)
        dummy = torch.randn(1, 3, args.imgsz, args.imgsz, dtype=torch.float32)
        with torch.no_grad():
            before_preparation = (
                normalize_torch_prediction(model(dummy)).detach().cpu().clone()
            )

        preparation_changes = prepare_model_for_vitis_inspection(model, torch.nn.SiLU, C2f)
        with torch.no_grad():
            after_preparation = (
                normalize_torch_prediction(model(dummy)).detach().cpu().clone()
            )
        preparation_parity = compare_prediction_tensors(
            torch,
            before_preparation,
            after_preparation,
            rtol=MODEL_PREPARATION_RTOL,
            atol=MODEL_PREPARATION_ATOL,
        )
        payload["model_preparation"] = {
            "changes": preparation_changes,
            "prediction_parity": preparation_parity,
        }
        if not preparation_parity["allclose"]:
            raise ModelPreparationError(
                "Vitis graph preparation changed raw predictions; inspect model_preparation in the manifest."
            )

        Inspector(args.target).inspect(
            model,
            (dummy,),
            device=torch.device("cpu"),
            output_dir=str(output_dir),
            verbose_level=args.verbose_level,
            image_format=None if args.image_format == "none" else args.image_format,
        )

        payload["status"] = "ok"
        payload["completed_utc"] = common.utc_now()
        common.write_json_atomic(manifest, payload, overwrite=args.overwrite)
        print(f"Inspector output: {output_dir}")
        print(f"Inspection manifest: {manifest}")
        return 0
    except Exception as exc:
        payload["status"] = "failed"
        payload["completed_utc"] = common.utc_now()
        payload["error"] = {"type": type(exc).__name__, "message": str(exc)}
        common.write_json_atomic(manifest, payload, overwrite=args.overwrite)
        raise


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except (
        common.DependencyError,
        FileExistsError,
        FileNotFoundError,
        ModelPreparationError,
        ValueError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
