#!/usr/bin/env python3
"""Run Vitis AI Inspector on an FSWD-YOLO PyTorch checkpoint."""

from __future__ import annotations

import argparse
import inspect
import os
import platform
import sys
from pathlib import Path
from types import MethodType
from typing import Any, Dict, Optional, Sequence, Tuple

try:
    from scripts import fswd_deploy_common as common
    from scripts import fswd_vitis_report as vitis_report
except ImportError:  # Direct execution: python scripts/inspect_fswd_vitis.py
    import fswd_deploy_common as common
    import fswd_vitis_report as vitis_report


REPO_ROOT = common.add_repo_root_to_path(Path(__file__))
MODEL_PREPARATION_RTOL = 1e-5
MODEL_PREPARATION_ATOL = 1e-6


class ModelPreparationError(RuntimeError):
    """Raised when a Vitis graph preparation changes model predictions."""


def activation_mode_metadata(activation_experiment: str) -> Dict[str, Any]:
    """Describe whether this run audits the checkpoint or changes activation semantics."""
    if activation_experiment not in {"none", "hardswish", "hardsigmoid"}:
        raise ValueError(f"unsupported activation experiment: {activation_experiment}")
    is_audit = activation_experiment == "none"
    return {
        "mode": "original_model_audit" if is_audit else "deployment_activation_experiment",
        "activation_experiment": activation_experiment,
        "semantic_equivalence_required": is_audit,
        "checkpoint_modified": False,
    }


def prepare_model_for_vitis_inspection(
    model: Any,
    silu_class: Any,
    hardswish_class: Any = None,
) -> Dict[str, int]:
    """Apply semantics-preserving graph forms preferred by Vitis AI Inspector."""
    changes = {
        "silu_inplace_disabled": 0,
        "hardswish_inplace_disabled": 0,
        "c2f_forward_split_enabled": 0,
    }
    for module in model.modules():
        if isinstance(module, silu_class) and getattr(module, "inplace", False):
            module.inplace = False
            changes["silu_inplace_disabled"] += 1
        if (
            hardswish_class is not None
            and isinstance(module, hardswish_class)
            and getattr(module, "inplace", False)
        ):
            module.inplace = False
            changes["hardswish_inplace_disabled"] += 1

    return changes


def apply_activation_experiment(
    model: Any,
    silu_class: Any,
    activation_name: str,
    forward_function: Any,
) -> Dict[str, int]:
    """Bind an explicit non-equivalent hard activation to SiLU modules in memory."""
    if activation_name not in {"hardswish", "hardsigmoid"}:
        raise ValueError("activation experiment requires an explicit hard activation")

    replaced = 0
    for module in model.modules():
        if isinstance(module, silu_class):
            module.forward = MethodType(forward_function, module)
            replaced += 1
    return {"silu_activation_replaced": replaced}


def make_activation_forward(functional: Any, activation_name: str) -> Any:
    """Build the unbound module forward used by a hard-activation experiment."""
    if activation_name not in {"hardswish", "hardsigmoid"}:
        raise ValueError("activation experiment requires an explicit hard activation")
    activation = getattr(functional, activation_name)

    def forward(_module: Any, value: Any) -> Any:
        return activation(value)

    return forward


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


def compare_prediction_sequences(
    torch_module: Any,
    before: Sequence[Any],
    after: Sequence[Any],
    rtol: float,
    atol: float,
) -> Dict[str, Any]:
    """Compare one or more prediction tensors without hiding output-count changes."""
    output_count_matches = len(before) == len(after)
    comparisons = []
    if output_count_matches:
        comparisons = [
            compare_prediction_tensors(torch_module, before_value, after_value, rtol, atol)
            for before_value, after_value in zip(before, after)
        ]
    shape_matches = output_count_matches and all(item["shape_matches"] for item in comparisons)
    allclose = shape_matches and all(item["allclose"] for item in comparisons)
    maximums = [item["max_absolute_error"] for item in comparisons if item["max_absolute_error"] is not None]
    means = [item["mean_absolute_error"] for item in comparisons if item["mean_absolute_error"] is not None]
    return {
        "before_shapes": [list(value.shape) for value in before],
        "after_shapes": [list(value.shape) for value in after],
        "output_count_matches": output_count_matches,
        "shape_matches": shape_matches,
        "allclose": allclose,
        "max_absolute_error": max(maximums) if maximums else None,
        "mean_absolute_error": sum(means) / len(means) if means else None,
        "outputs": comparisons,
        "rtol": rtol,
        "atol": atol,
    }


def enforce_prediction_comparison(
    comparison: Dict[str, Any], semantic_equivalence_required: bool
) -> None:
    """Reject shape changes and unexpected drift in original-model audit mode."""
    if not comparison["shape_matches"]:
        raise ModelPreparationError(
            "Vitis graph preparation changed the raw prediction shape."
        )
    if semantic_equivalence_required and not comparison["allclose"]:
        raise ModelPreparationError(
            "Vitis graph preparation changed raw predictions; inspect model_preparation in the manifest."
        )


def build_model_preparation_payload(
    activation_experiment: str,
    changes: Dict[str, int],
    prediction_comparison: Dict[str, Any],
) -> Dict[str, Any]:
    """Combine stable mode metadata with changes and prediction evidence."""
    return {
        **activation_mode_metadata(activation_experiment),
        "changes": changes,
        "prediction_comparison": prediction_comparison,
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
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--weights", type=Path, help="Input Ultralytics .pt checkpoint")
    source.add_argument("--model-config", type=Path, help="Input Ultralytics YAML model candidate")
    parser.add_argument(
        "--target",
        required=True,
        help="Vitis AI Inspector target name or fingerprint for the intended hardware",
    )
    parser.add_argument(
        "--raw-detect-output",
        action="store_true",
        help="Inspect only convolutional Detect outputs; decode and NMS remain on the host",
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
    parser.add_argument(
        "--activation-experiment",
        choices=("none", "hardswish", "hardsigmoid"),
        default="none",
        help="Explicit non-equivalent activation graph experiment (default: none)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Inspector input seed (default: 0)")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing manifest")
    return parser


def validate_source_arguments(args: argparse.Namespace) -> Tuple[str, Path]:
    """Validate the selected model source and candidate-only restrictions."""
    if args.model_config is not None:
        if args.activation_experiment != "none":
            raise ValueError("A YAML candidate cannot be combined with --activation-experiment")
        return "model_config", common.validate_model_config(args.model_config)
    return "weights", common.validate_weights(args.weights)


def model_source_metadata(source_kind: str, source_path: Path) -> Dict[str, str]:
    """Describe whether Inspector is evaluating trained weights or a random candidate."""
    if source_kind == "weights":
        kind = "trained_checkpoint"
        weight_state = "trained"
    elif source_kind == "model_config":
        kind = "model_config_candidate"
        weight_state = "random_initialization"
    else:
        raise ValueError(f"unsupported model source kind: {source_kind}")
    return {
        "kind": kind,
        "path": str(source_path),
        "sha256": common.sha256_file(source_path),
        "weight_state": weight_state,
    }


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


def _base_manifest(
    args: argparse.Namespace,
    source_kind: str,
    source_path: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    source_metadata = model_source_metadata(source_kind, source_path)
    return {
        "tool": "inspect_fswd_vitis",
        "status": "running",
        "timestamp_utc": common.utc_now(),
        "arguments": _serialize_arguments(args),
        "target": args.target,
        "input": {
            "model_source": source_metadata,
            source_kind: str(source_path),
            f"{source_kind}_sha256": source_metadata["sha256"],
            "tensor_shape": [1, 3, args.imgsz, args.imgsz],
            "dtype": "float32",
            "device": "cpu",
            "output_contract": (
                "raw_detect_feature_maps" if args.raw_detect_output else "decoded_predictions"
            ),
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

    source_kind, source_path = validate_source_arguments(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    manifest = prepare_manifest(output_dir, args.overwrite)
    payload = _base_manifest(args, source_kind, source_path, output_dir)

    os.environ["YOLO_AUTOINSTALL"] = "false"
    try:
        common.require_modules(inspector_requirements(), "FSWD-YOLO Vitis AI inspection")

        import torch

        common.normalize_version_attribute(torch)
        from nndct_shared.base import NNDCT_OP
        from pytorch_nndct.apis import Inspector
        from pytorch_nndct.hardware_v3.inspector import InspectorImpl
        from ultralytics import YOLO
        from scripts.fswd_dpu_model import RawDetectHeadAdapter
        from scripts.export_fswd_onnx import normalize_torch_prediction

        compatibility_applied = install_vitis_35_permute_report_compatibility(
            InspectorImpl, NNDCT_OP.PERMUTE
        )
        payload["compatibility"] = {
            "vitis_35_permute_report_patch_applied": compatibility_applied,
            "scope": "Inspector report messages only",
        }

        model = YOLO(str(source_path)).model.float().eval().cpu()
        inspection_model = RawDetectHeadAdapter(model).eval() if args.raw_detect_output else model
        torch.manual_seed(args.seed)
        dummy = torch.randn(1, 3, args.imgsz, args.imgsz, dtype=torch.float32)

        def normalized_outputs(output: Any) -> Tuple[Any, ...]:
            if args.raw_detect_output:
                if not isinstance(output, (list, tuple)) or not all(
                    isinstance(value, torch.Tensor) for value in output
                ):
                    raise TypeError("raw Detect inspection requires a tensor sequence")
                return tuple(value.detach().cpu().clone() for value in output)
            return (normalize_torch_prediction(output).detach().cpu().clone(),)

        with torch.no_grad():
            before_preparation = normalized_outputs(inspection_model(dummy))

        preparation_changes = prepare_model_for_vitis_inspection(
            model, torch.nn.SiLU, torch.nn.Hardswish
        )
        mode = activation_mode_metadata(args.activation_experiment)
        if args.activation_experiment == "none":
            preparation_changes["silu_activation_replaced"] = 0
        else:
            preparation_changes.update(
                apply_activation_experiment(
                    model,
                    torch.nn.SiLU,
                    args.activation_experiment,
                    make_activation_forward(torch.nn.functional, args.activation_experiment),
                )
            )
        with torch.no_grad():
            after_preparation = normalized_outputs(inspection_model(dummy))
        preparation_parity = compare_prediction_sequences(
            torch,
            before_preparation,
            after_preparation,
            rtol=MODEL_PREPARATION_RTOL,
            atol=MODEL_PREPARATION_ATOL,
        )
        payload["model_preparation"] = build_model_preparation_payload(
            args.activation_experiment,
            preparation_changes,
            preparation_parity,
        )
        enforce_prediction_comparison(
            preparation_parity,
            semantic_equivalence_required=mode["semantic_equivalence_required"],
        )
        if not mode["semantic_equivalence_required"]:
            print(
                "WARNING: This is a non-equivalent deployment activation experiment. "
                "Its Inspector result does not establish retained model accuracy; "
                "retraining or fine-tuning is required."
            )

        reports_before = vitis_report.snapshot_report_signatures(output_dir)
        Inspector(args.target).inspect(
            inspection_model,
            (dummy,),
            device=torch.device("cpu"),
            output_dir=str(output_dir),
            verbose_level=args.verbose_level,
            image_format=None if args.image_format == "none" else args.image_format,
        )
        report_path = vitis_report.find_generated_report(output_dir, reports_before)
        payload["inspection_summary"] = vitis_report.summarize_inspector_report(
            report_path
        )

        payload["status"] = "ok"
        payload["completed_utc"] = common.utc_now()
        common.write_json_atomic(manifest, payload, overwrite=args.overwrite)
        print(f"Inspector output: {output_dir}")
        print(f"Inspection manifest: {manifest}")
        print(
            "Unique CPU findings: "
            f"{payload['inspection_summary']['unique_row_count']}"
        )
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
        vitis_report.InspectorReportError,
        ValueError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
