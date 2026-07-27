#!/usr/bin/env python3
"""Initialize an FSWD-YOLO DPU checkpoint from an existing trained checkpoint."""

from __future__ import annotations

import argparse
import copy
import os
import platform
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

try:
    from scripts import fswd_deploy_common as common
except ImportError:  # Direct execution
    import fswd_deploy_common as common


REPO_ROOT = common.add_repo_root_to_path(Path(__file__))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Initialize a split-free FSWD-YOLO DPU checkpoint without installing dependencies."
    )
    parser.add_argument("--source-checkpoint", required=True, type=Path, help="Input trained .pt checkpoint")
    parser.add_argument("--model-config", required=True, type=Path, help="DPU candidate YAML model config")
    parser.add_argument("--output", required=True, type=Path, help="Output initialized .pt checkpoint")
    parser.add_argument("--report", type=Path, help="Migration JSON report path")
    parser.add_argument("--imgsz", type=int, default=640, help="Square profile/parity input size (default: 640)")
    parser.add_argument("--seed", type=int, default=0, help="Decode parity input seed (default: 0)")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output and report")
    return parser


def migration_report_path(output: Path) -> Path:
    """Return the default sidecar report path for an initialized checkpoint."""
    return Path(f"{output}.migration.json")


def validate_arguments(args: argparse.Namespace) -> Tuple[Path, Path]:
    """Validate source paths and scalar arguments."""
    if args.imgsz <= 0:
        raise ValueError("--imgsz must be greater than zero")
    return common.validate_weights(args.source_checkpoint), common.validate_model_config(args.model_config)


def prepare_artifacts(output: Path, report: Path, overwrite: bool) -> Tuple[Path, Path]:
    """Validate both generated artifact paths before writing either one."""
    output_path = common.prepare_output(output, overwrite=overwrite, expected_suffix=".pt")
    report_path = common.prepare_output(report, overwrite=overwrite, expected_suffix=".json")
    if output_path == report_path:
        raise ValueError("Checkpoint and report paths must be different")
    return output_path, report_path


def initialize_requirements() -> Dict[str, str]:
    return {
        "torch": "activate the PyTorch environment used by this FSWD-YOLO repository",
        "ultralytics": "run from this repository or install it in the active environment",
    }


def migration_semantics() -> Dict[str, Any]:
    """Describe which migration transformations are and are not functionally equivalent."""
    return {
        "source_to_candidate_fp32_equivalent": False,
        "reason": (
            "The destination intentionally replaces SiLU and C2PSFCA; successful state mapping "
            "does not imply source-model prediction parity."
        ),
        "gsconv_shuffle_rewrite": "exact_when_activation_modules_match",
    }


def _serialize_arguments(args: argparse.Namespace) -> Dict[str, Any]:
    return {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()}


def _model_metrics(model: Any, imgsz: int, get_flops: Any) -> Dict[str, Any]:
    parameters = sum(parameter.numel() for parameter in model.parameters())
    deployment_tensors = sum(tensor.numel() for tensor in model.state_dict().values())
    flops = float(get_flops(model, imgsz))
    return {
        "parameters": int(parameters),
        "persistent_buffers": int(deployment_tensors - parameters),
        "deployment_tensors": int(deployment_tensors),
        "gflops": flops,
        "gflops_available": flops > 0.0,
    }


def _resource_comparison(source: Dict[str, Any], destination: Dict[str, Any]) -> Dict[str, Any]:
    source_deployment = source.get("deployment_tensors", source["parameters"])
    destination_deployment = destination.get("deployment_tensors", destination["parameters"])
    parameter_ratio = destination_deployment / source_deployment if source_deployment else None
    learned_parameter_ratio = (
        destination["parameters"] / source["parameters"] if source["parameters"] else None
    )
    flop_ratio = (
        destination["gflops"] / source["gflops"]
        if source["gflops_available"] and destination["gflops_available"]
        else None
    )
    return {
        "source": source,
        "destination": destination,
        "parameter_ratio": parameter_ratio,
        "learned_parameter_ratio": learned_parameter_ratio,
        "parameter_basis": "deployment_tensors",
        "flop_ratio": flop_ratio,
        "parameter_gate": parameter_ratio is not None and parameter_ratio <= 1.10,
        "flop_gate": flop_ratio is not None and flop_ratio <= 1.0,
    }


def _save_checkpoint_atomic(torch_module: Any, checkpoint: Dict[str, Any], output: Path) -> None:
    temporary_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
        torch_module.save(checkpoint, temporary_path)
        os.replace(temporary_path, output)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _base_report(
    args: argparse.Namespace,
    source_checkpoint: Path,
    model_config: Path,
    output: Path,
    report: Path,
) -> Dict[str, Any]:
    return {
        "tool": "initialize_fswd_dpu",
        "status": "running",
        "timestamp_utc": common.utc_now(),
        "arguments": _serialize_arguments(args),
        "source": {
            "checkpoint": str(source_checkpoint),
            "checkpoint_sha256": common.sha256_file(source_checkpoint),
            "model_config": str(model_config),
            "model_config_sha256": common.sha256_file(model_config),
        },
        "output": {"checkpoint": str(output), "report": str(report)},
        "git": common.git_metadata(REPO_ROOT),
        "runtime": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
        },
        "packages": common.package_versions(["torch", "ultralytics", "thop"]),
        "migration_semantics": migration_semantics(),
    }


def run(args: argparse.Namespace) -> int:
    source_checkpoint, model_config = validate_arguments(args)
    requested_report = args.report if args.report is not None else migration_report_path(args.output)
    output, report = prepare_artifacts(args.output, requested_report, args.overwrite)
    payload = _base_report(args, source_checkpoint, model_config, output, report)

    os.environ["YOLO_AUTOINSTALL"] = "false"
    try:
        common.require_modules(initialize_requirements(), "FSWD-YOLO DPU checkpoint initialization")

        import torch

        common.normalize_version_attribute(torch)
        from scripts.fswd_dpu_migration import migrate_fswd_model
        from scripts.fswd_dpu_model import compare_decode_parity
        from ultralytics import YOLO, __version__
        from ultralytics.nn.tasks import attempt_load_one_weight
        from ultralytics.utils.torch_utils import get_flops

        source_model, source_payload = attempt_load_one_weight(str(source_checkpoint), device="cpu", fuse=False)
        source_model = source_model.float().eval().cpu()
        destination_model = YOLO(str(model_config)).model.float().eval().cpu()
        if hasattr(source_model, "names"):
            destination_model.names = copy.deepcopy(source_model.names)

        migration = migrate_fswd_model(source_model, destination_model)
        torch.manual_seed(args.seed)
        dummy = torch.randn(1, 3, args.imgsz, args.imgsz, dtype=torch.float32)
        decode_parity = compare_decode_parity(destination_model, dummy)
        if not decode_parity["shape_matches"] or not decode_parity["allclose"]:
            raise RuntimeError("Raw Detect host decoding does not match native destination-model decoding")

        source_metrics = _model_metrics(source_model, args.imgsz, get_flops)
        destination_metrics = _model_metrics(destination_model, args.imgsz, get_flops)
        resources = _resource_comparison(source_metrics, destination_metrics)

        checkpoint = dict(source_payload)
        checkpoint.update(
            {
                "epoch": -1,
                "best_fitness": None,
                "model": copy.deepcopy(destination_model).half(),
                "ema": None,
                "updates": None,
                "optimizer": None,
                "date": common.utc_now(),
                "version": __version__,
                "license": "AGPL-3.0 (https://ultralytics.com/license)",
                "docs": "https://docs.ultralytics.com",
            }
        )
        _save_checkpoint_atomic(torch, checkpoint, output)

        payload["migration"] = migration
        payload["decode_parity"] = decode_parity
        payload["resources"] = resources
        payload["output"]["checkpoint_sha256"] = common.sha256_file(output)
        payload["status"] = "ok"
        payload["completed_utc"] = common.utc_now()
        common.write_json_atomic(report, payload, overwrite=True)
        print(f"Initialized checkpoint: {output}")
        print(f"Migration report: {report}")
        return 0
    except Exception as exc:
        payload["status"] = "failed"
        payload["completed_utc"] = common.utc_now()
        payload["error"] = {"type": type(exc).__name__, "message": str(exc)}
        common.write_json_atomic(report, payload, overwrite=True)
        raise


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except (
        common.DependencyError,
        FileExistsError,
        FileNotFoundError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
