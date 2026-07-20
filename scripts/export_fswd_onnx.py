#!/usr/bin/env python3
"""Export an FSWD-YOLO checkpoint to a reviewable, fixed-shape ONNX model."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Sequence

try:
    from scripts import fswd_deploy_common as common
except ImportError:  # Direct execution: python scripts/export_fswd_onnx.py
    import fswd_deploy_common as common


SUPPORTED_ONNX = "onnx>=1.12.0,<1.18.0"


class RuntimeParityError(RuntimeError):
    """Raised when PyTorch and ONNX Runtime outputs do not agree."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export FSWD-YOLO to fixed-shape FP32 ONNX and validate the graph."
    )
    parser.add_argument("--weights", required=True, type=Path, help="Input Ultralytics .pt checkpoint")
    parser.add_argument("--output", required=True, type=Path, help="Destination .onnx path")
    parser.add_argument("--imgsz", type=int, default=640, help="Square input size (default: 640)")
    parser.add_argument("--opset", type=int, default=13, help="ONNX opset (default: 13)")
    parser.add_argument(
        "--verify-runtime",
        action="store_true",
        help="Compare deterministic PyTorch and ONNX Runtime outputs",
    )
    parser.add_argument("--seed", type=int, default=0, help="Parity input seed (default: 0)")
    parser.add_argument("--rtol", type=float, default=1e-4, help="Parity relative tolerance")
    parser.add_argument("--atol", type=float, default=1e-5, help="Parity absolute tolerance")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing artifacts")
    return parser


def export_requirements(verify_runtime: bool) -> Dict[str, str]:
    requirements = {
        "torch": "install the PyTorch version used to train the checkpoint",
        "onnx": f"install {SUPPORTED_ONNX}",
        "ultralytics": "install this repository in the active Python environment",
    }
    if verify_runtime:
        requirements.update(
            {
                "numpy": "install a NumPy version compatible with PyTorch and ONNX Runtime",
                "onnxruntime": "install onnxruntime to enable --verify-runtime",
            }
        )
    return requirements


def report_path(output: Path) -> Path:
    return output.with_name(f"{output.name}.report.json")


def _numeric_version(version: str) -> tuple[int, int, int]:
    match = re.match(r"^\s*(\d+)(?:\.(\d+))?(?:\.(\d+))?", version)
    if match is None:
        raise common.DependencyError(
            f"Cannot parse installed ONNX version {version!r}; expected {SUPPORTED_ONNX}."
        )
    return tuple(int(part or 0) for part in match.groups())


def validate_onnx_version(version: str) -> None:
    parsed = _numeric_version(version)
    if parsed < (1, 12, 0) or parsed >= (1, 18, 0):
        raise common.DependencyError(
            f"Unsupported ONNX version {version}; expected {SUPPORTED_ONNX}. "
            "This tool does not change the active environment."
        )


def export_kwargs(imgsz: int, opset: int) -> Dict[str, Any]:
    return {
        "format": "onnx",
        "imgsz": imgsz,
        "batch": 1,
        "dynamic": False,
        "nms": False,
        "simplify": False,
        "half": False,
        "opset": opset,
        "device": "cpu",
    }


def normalize_torch_prediction(value: Any) -> Any:
    """Find the first tensor-like prediction in common Ultralytics return structures."""
    if hasattr(value, "detach") and hasattr(value, "shape"):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            try:
                return normalize_torch_prediction(item)
            except TypeError:
                continue
    raise TypeError("Could not locate a prediction tensor in the PyTorch model output.")


def _dimension_value(dimension: Any) -> Any:
    if dimension.HasField("dim_value"):
        return dimension.dim_value
    if dimension.HasField("dim_param"):
        return dimension.dim_param
    return None


def _value_info(value: Any, onnx: Any) -> Dict[str, Any]:
    tensor_type = value.type.tensor_type
    return {
        "name": value.name,
        "element_type": onnx.TensorProto.DataType.Name(tensor_type.elem_type),
        "shape": [_dimension_value(dimension) for dimension in tensor_type.shape.dim],
    }


def onnx_graph_report(model: Any, onnx: Any) -> Dict[str, Any]:
    return {
        "ir_version": model.ir_version,
        "opset_imports": [
            {"domain": item.domain or "ai.onnx", "version": item.version}
            for item in model.opset_import
        ],
        "inputs": [_value_info(value, onnx) for value in model.graph.input],
        "outputs": [_value_info(value, onnx) for value in model.graph.output],
        "node_count": len(model.graph.node),
        "operators": common.summarize_operators(node.op_type for node in model.graph.node),
    }


def run_runtime_parity(
    yolo: Any,
    output: Path,
    imgsz: int,
    seed: int,
    rtol: float,
    atol: float,
) -> Dict[str, Any]:
    import numpy as np
    import onnxruntime as ort
    import torch

    torch.manual_seed(seed)
    model = yolo.model.float().eval().cpu()
    sample = torch.randn(1, 3, imgsz, imgsz, dtype=torch.float32)
    with torch.no_grad():
        torch_output = normalize_torch_prediction(model(sample)).detach().cpu().numpy()

    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    ort_output = session.run(None, {session.get_inputs()[0].name: sample.numpy()})[0]
    shape_matches = torch_output.shape == ort_output.shape
    if shape_matches:
        absolute_error = np.abs(torch_output - ort_output)
        max_absolute_error = float(absolute_error.max(initial=0.0))
        mean_absolute_error = float(absolute_error.mean())
        allclose = bool(np.allclose(torch_output, ort_output, rtol=rtol, atol=atol))
    else:
        max_absolute_error = None
        mean_absolute_error = None
        allclose = False

    return {
        "seed": seed,
        "rtol": rtol,
        "atol": atol,
        "torch_shape": list(torch_output.shape),
        "onnxruntime_shape": list(ort_output.shape),
        "shape_matches": shape_matches,
        "allclose": allclose,
        "max_absolute_error": max_absolute_error,
        "mean_absolute_error": mean_absolute_error,
    }


def _serialize_arguments(args: argparse.Namespace) -> Dict[str, Any]:
    return {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()}


def _move_exported_artifact(source: Path, destination: Path) -> None:
    if source.resolve() == destination.resolve():
        return
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
        shutil.copy2(source, temporary_path)
        os.replace(temporary_path, destination)
        source.unlink()
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _base_report(args: argparse.Namespace, weights: Path) -> Dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    return {
        "tool": "export_fswd_onnx",
        "status": "running",
        "timestamp_utc": common.utc_now(),
        "arguments": _serialize_arguments(args),
        "input": {"path": str(weights), "sha256": common.sha256_file(weights)},
        "git": common.git_metadata(repo_root),
        "packages": common.package_versions(
            ["torch", "onnx", "onnxruntime", "ultralytics", "numpy"]
        ),
    }


def run(args: argparse.Namespace) -> int:
    if args.imgsz <= 0:
        raise ValueError("--imgsz must be greater than zero")
    if args.opset <= 0:
        raise ValueError("--opset must be greater than zero")
    if args.rtol < 0 or args.atol < 0:
        raise ValueError("--rtol and --atol must be non-negative")

    weights = common.validate_weights(args.weights)
    output = common.prepare_output(args.output, args.overwrite)
    report = report_path(output)
    common.prepare_output(report, args.overwrite, expected_suffix=".json")

    generated_default = weights.with_suffix(".onnx")
    if generated_default.resolve() != output.resolve():
        common.prepare_output(generated_default, args.overwrite)

    common.require_modules(export_requirements(args.verify_runtime), "FSWD-YOLO ONNX export")
    payload = _base_report(args, weights)

    # Ultralytics reads this at import time. The deployment tools never install packages.
    os.environ["YOLO_AUTOINSTALL"] = "false"
    try:
        import onnx
        from ultralytics import YOLO

        validate_onnx_version(onnx.__version__)
        yolo = YOLO(str(weights))
        exported = Path(yolo.export(**export_kwargs(args.imgsz, args.opset))).expanduser().resolve()
        if not exported.is_file():
            raise FileNotFoundError(f"Ultralytics did not produce the reported ONNX file: {exported}")
        _move_exported_artifact(exported, output)

        model = onnx.load(str(output))
        onnx.checker.check_model(model)
        payload["onnx"] = onnx_graph_report(model, onnx)
        payload["output"] = {"path": str(output), "sha256": common.sha256_file(output)}

        if args.verify_runtime:
            payload["runtime_parity"] = run_runtime_parity(
                yolo, output, args.imgsz, args.seed, args.rtol, args.atol
            )
            if not payload["runtime_parity"]["allclose"]:
                raise RuntimeParityError(
                    "PyTorch and ONNX Runtime outputs differ; inspect runtime_parity in the report."
                )

        payload["status"] = "ok"
        payload["completed_utc"] = common.utc_now()
        common.write_json_atomic(report, payload, overwrite=args.overwrite)
        print(f"ONNX export: {output}")
        print(f"Validation report: {report}")
        return 0
    except Exception as exc:
        payload["status"] = "failed"
        payload["completed_utc"] = common.utc_now()
        payload["error"] = {"type": type(exc).__name__, "message": str(exc)}
        common.write_json_atomic(report, payload, overwrite=args.overwrite)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except RuntimeParityError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except (common.DependencyError, FileExistsError, FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
