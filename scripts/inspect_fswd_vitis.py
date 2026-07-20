#!/usr/bin/env python3
"""Run Vitis AI Inspector on an FSWD-YOLO PyTorch checkpoint."""

from __future__ import annotations

import argparse
import os
import platform
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

try:
    from scripts import fswd_deploy_common as common
except ImportError:  # Direct execution: python scripts/inspect_fswd_vitis.py
    import fswd_deploy_common as common


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
    repo_root = Path(__file__).resolve().parents[1]
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
        "git": common.git_metadata(repo_root),
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
        from pytorch_nndct.apis import Inspector
        from ultralytics import YOLO

        torch.manual_seed(args.seed)
        model = YOLO(str(weights)).model.float().eval().cpu()
        dummy = torch.randn(1, 3, args.imgsz, args.imgsz, dtype=torch.float32)
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
    except (common.DependencyError, FileExistsError, FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
