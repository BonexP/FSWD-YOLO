#!/usr/bin/env python3
"""Profile and gate the FSWD-YOLO DPU model candidate."""

from __future__ import annotations

import argparse
import os
import platform
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

try:
    from scripts import fswd_deploy_common as common
except ImportError:  # Direct execution
    import fswd_deploy_common as common


REPO_ROOT = common.add_repo_root_to_path(Path(__file__))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile FSWD-YOLO baseline and DPU candidate resource budgets.")
    parser.add_argument("--baseline-config", required=True, type=Path, help="Original FSWD-YOLO YAML")
    parser.add_argument("--candidate-config", required=True, type=Path, help="DPU candidate YAML")
    parser.add_argument("--output", required=True, type=Path, help="Output JSON report")
    parser.add_argument("--imgsz", type=int, default=640, help="Square profile input size (default: 640)")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing report")
    return parser


def profile_requirements() -> Dict[str, str]:
    return {
        "torch": "activate the PyTorch environment used by this FSWD-YOLO repository",
        "ultralytics": "run from this repository or install it in the active environment",
        "thop": "provide thop in the active environment for the mandatory FLOP gate",
    }


def evaluate_resource_gates(baseline: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate the approved parameter and FLOP limits."""
    if baseline["parameters"] <= 0 or baseline["gflops"] <= 0:
        raise ValueError("Baseline parameters and GFLOPs must be greater than zero")
    parameter_ratio = candidate["parameters"] / baseline["parameters"]
    flop_ratio = candidate["gflops"] / baseline["gflops"]
    return {
        "parameter_ratio": parameter_ratio,
        "flop_ratio": flop_ratio,
        "parameter_gate": parameter_ratio <= 1.10,
        "flop_gate": flop_ratio <= 1.0,
        "all_pass": parameter_ratio <= 1.10 and flop_ratio <= 1.0,
    }


def _module_type_parameters(model: Any) -> Dict[str, int]:
    counts: Counter[str] = Counter()
    for module in model.modules():
        counts[type(module).__name__] += sum(
            parameter.numel() for parameter in module.parameters(recurse=False)
        )
    return dict(sorted(counts.items()))


def _profile_model(model: Any, imgsz: int, get_flops: Any) -> Dict[str, Any]:
    parameters = sum(parameter.numel() for parameter in model.parameters())
    gflops = float(get_flops(model, imgsz))
    if gflops <= 0.0:
        raise RuntimeError("FLOP profiling returned zero; verify thop and the model forward path")
    return {
        "parameters": int(parameters),
        "gflops": gflops,
        "parameters_by_module_type": _module_type_parameters(model),
    }


def _module_type_delta(baseline: Dict[str, int], candidate: Dict[str, int]) -> Dict[str, int]:
    names = sorted(set(baseline) | set(candidate))
    return {name: candidate.get(name, 0) - baseline.get(name, 0) for name in names}


def _serialize_arguments(args: argparse.Namespace) -> Dict[str, Any]:
    return {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()}


def run(args: argparse.Namespace) -> int:
    if args.imgsz <= 0:
        raise ValueError("--imgsz must be greater than zero")
    baseline_config = common.validate_model_config(args.baseline_config)
    candidate_config = common.validate_model_config(args.candidate_config)
    output = common.prepare_output(args.output, args.overwrite, expected_suffix=".json")
    payload: Dict[str, Any] = {
        "tool": "profile_fswd_dpu_candidate",
        "status": "running",
        "timestamp_utc": common.utc_now(),
        "arguments": _serialize_arguments(args),
        "sources": {
            "baseline": {
                "path": str(baseline_config),
                "sha256": common.sha256_file(baseline_config),
            },
            "candidate": {
                "path": str(candidate_config),
                "sha256": common.sha256_file(candidate_config),
            },
        },
        "git": common.git_metadata(REPO_ROOT),
        "runtime": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
        },
        "packages": common.package_versions(["torch", "ultralytics", "thop"]),
    }

    os.environ["YOLO_AUTOINSTALL"] = "false"
    try:
        common.require_modules(profile_requirements(), "FSWD-YOLO DPU resource profiling")

        import torch

        common.normalize_version_attribute(torch)
        from ultralytics import YOLO
        from ultralytics.utils.torch_utils import get_flops

        baseline_model = YOLO(str(baseline_config)).model.float().eval().cpu()
        candidate_model = YOLO(str(candidate_config)).model.float().eval().cpu()
        baseline = _profile_model(baseline_model, args.imgsz, get_flops)
        candidate = _profile_model(candidate_model, args.imgsz, get_flops)
        gates = evaluate_resource_gates(baseline, candidate)
        payload["baseline"] = baseline
        payload["candidate"] = candidate
        payload["module_type_parameter_delta"] = _module_type_delta(
            baseline["parameters_by_module_type"], candidate["parameters_by_module_type"]
        )
        payload["gates"] = gates
        payload["status"] = "ok" if gates["all_pass"] else "rejected"
        payload["completed_utc"] = common.utc_now()
        common.write_json_atomic(output, payload, overwrite=True)
        print(f"Resource report: {output}")
        print(f"Parameter gate: {'PASS' if gates['parameter_gate'] else 'FAIL'}")
        print(f"FLOP gate: {'PASS' if gates['flop_gate'] else 'FAIL'}")
        return 0 if gates["all_pass"] else 2
    except Exception as exc:
        payload["status"] = "failed"
        payload["completed_utc"] = common.utc_now()
        payload["error"] = {"type": type(exc).__name__, "message": str(exc)}
        common.write_json_atomic(output, payload, overwrite=True)
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
