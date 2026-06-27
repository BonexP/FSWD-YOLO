#!/usr/bin/env python3
"""Export model parameter counts and GFLOPs for revision experiments."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


MAIN_MODELS = [
    ("main", "yolov8s", "ultralytics/cfg/models/v8/yolov8s.yaml"),
    ("main", "yolov9s", "ultralytics/cfg/models/v9/yolov9s.yaml"),
    ("main", "yolov10s", "ultralytics/cfg/models/v10/yolov10s.yaml"),
    ("main", "yolo11s", "ultralytics/cfg/models/11/yolo11s.yaml"),
    ("main", "fswd_yolo", "ultralytics/cfg/models/11/fswd-yolo.yaml"),
]

WORKSET_MODELS = [
    ("main", "yolov8s", "ultralytics/cfg/models/v8/yolov8s.yaml"),
    ("main", "yolov9s", "ultralytics/cfg/models/v9/yolov9s.yaml"),
    ("main", "fswd_yolo", "ultralytics/cfg/models/11/fswd-yolo.yaml"),
]

ABLATION_MODELS = [
    ("ablation", "ghost_fca_ciou", "ultralytics/cfg/models/11/yolo11s_C3k2Ghost_C2PSFCA.yaml"),
    (
        "ablation",
        "ghostsimam_fca_ciou",
        "ultralytics/cfg/models/11/yolo11s_C3k2GhostSimAMinner_C2PSFCA.yaml",
    ),
    ("ablation", "ghost_vov_ciou", "ultralytics/cfg/models/11/yolo11s_C3k2Ghost_VoVCsingle.yaml"),
    (
        "ablation",
        "ghostsimam_vov_ciou",
        "ultralytics/cfg/models/11/yolo11s_C3k2GhostSimAMinner_VoVCsingle.yaml",
    ),
    ("ablation", "fca_vov_ciou", "ultralytics/cfg/models/11/yolo11s_C2PSFCA_VoVCsingle.yaml"),
    (
        "ablation",
        "ghost_fca_vov_ciou",
        "ultralytics/cfg/models/11/yolo11s_C3k2Ghost_C2PSFCA_VoVCsingle.yaml",
    ),
    (
        "ablation",
        "ghost_fca_vov_shapeiou",
        "ultralytics/cfg/models/11/yolo11s_C3k2Ghost_C2PSFCA_VoVCsingle.yaml",
    ),
    (
        "ablation",
        "ghostsimam_fca_vov_ciou_noweighted",
        "ultralytics/cfg/models/11/yolo11s_C3k2GhostSimAMinner_C2PSFCA_VoVCsingle.yaml",
    ),
    (
        "ablation",
        "ghostsimam_fca_vov_ciou",
        "ultralytics/cfg/models/11/yolo11s_C3k2GhostSimAMinner_C2PSFCA_VoVCsingle.yaml",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile trained revision weights or model YAML files and export "
            "parameters/GFLOPs tables for manuscript ablation analysis."
        )
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=None,
        help="Experiment root containing */weights/best.pt. Preferred for paper tables.",
    )
    parser.add_argument(
        "--preset",
        choices=["main", "workset", "ablation", "all"],
        default=None,
        help="Profile built-in revision model YAML presets when --runs-root is not used.",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Additional model YAML or weight path. Can be passed multiple times.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("revision_results/model_complexity"))
    parser.add_argument("--expected-count", type=int, default=0)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--device", default="cpu", help="Device for profiling, e.g. cpu or 0.")
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=None,
        help="Optional overall_mean_std.csv to merge with complexity rows by study/model.",
    )
    parser.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Profile every discovered run instead of profiling one representative per model.",
    )
    parser.add_argument("--dry-run", action="store_true", help="List profiling inputs without loading models.")
    return parser.parse_args()


def parse_run_name(run_name: str) -> Dict[str, str]:
    match = re.fullmatch(r"(?P<study>main|ablation|abl)_(?P<model>.+)_seed(?P<seed>[0-9]+)", run_name)
    if not match:
        return {"study": "unknown", "model": run_name, "seed": ""}
    meta = match.groupdict()
    if meta["study"] == "abl":
        meta["study"] = "ablation"
    return meta


def read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml

        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def discover_run_items(runs_root: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for weight in sorted(runs_root.glob("*/weights/best.pt"), key=lambda p: p.parent.parent.name):
        run_dir = weight.parent.parent
        meta = parse_run_name(run_dir.name)
        args_yaml = read_yaml(run_dir / "args.yaml")
        items.append(
            {
                "run": run_dir.name,
                "study": meta["study"],
                "model": meta["model"],
                "seed": meta["seed"],
                "source_type": "weights",
                "source": str(weight),
                "weights": str(weight),
                "model_config": str(args_yaml.get("model", "")),
                "args_yaml": str(run_dir / "args.yaml") if (run_dir / "args.yaml").exists() else "",
                "train_imgsz": args_yaml.get("imgsz", args_yaml.get("img_size", "")),
                "train_data": str(args_yaml.get("data", "")),
            }
        )
    return items


def preset_items(preset: str) -> List[Dict[str, Any]]:
    selected: List[tuple[str, str, str]] = []
    if preset in ("main", "all"):
        selected.extend(MAIN_MODELS)
    if preset in ("workset",):
        selected.extend(WORKSET_MODELS)
    if preset in ("ablation", "all"):
        selected.extend(ABLATION_MODELS)

    items: List[Dict[str, Any]] = []
    for study, model, cfg in selected:
        items.append(
            {
                "run": f"{study}_{model}",
                "study": study,
                "model": model,
                "seed": "",
                "source_type": "yaml",
                "source": cfg,
                "weights": "",
                "model_config": cfg,
                "args_yaml": "",
                "train_imgsz": "",
                "train_data": "",
            }
        )
    return items


def explicit_model_items(models: Sequence[str]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for spec in models:
        if "=" not in spec:
            raise SystemExit(f"--model must use NAME=PATH format, got: {spec}")
        name, path = spec.split("=", 1)
        source = Path(path)
        items.append(
            {
                "run": name,
                "study": "manual",
                "model": name,
                "seed": "",
                "source_type": "weights" if source.suffix == ".pt" else "yaml",
                "source": path,
                "weights": path if source.suffix == ".pt" else "",
                "model_config": path if source.suffix in {".yaml", ".yml"} else "",
                "args_yaml": "",
                "train_imgsz": "",
                "train_data": "",
            }
        )
    return items


def check_expected_count(items: Sequence[Dict[str, Any]], expected_count: int) -> None:
    if expected_count > 0 and len(items) != expected_count:
        raise SystemExit(f"Expected {expected_count} profiling inputs, found {len(items)}.")


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def move_to_device(model: Any, device_arg: str) -> str:
    import torch

    if device_arg == "cpu":
        device = torch.device("cpu")
    elif device_arg.isdigit():
        device = torch.device(f"cuda:{device_arg}" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_arg)
    model.to(device)
    model.eval()
    return str(device)


def count_layers(model: Any) -> int:
    layers = getattr(model, "model", None)
    if isinstance(layers, (list, tuple)):
        return len(layers)
    try:
        return sum(1 for _ in model.modules())
    except Exception:
        return 0


def compute_gflops(model: Any, img_size: int) -> tuple[float | None, str, str]:
    try:
        from ultralytics.utils.torch_utils import get_flops

        gflops = as_float(get_flops(model, imgsz=img_size))
        if gflops and gflops > 0:
            return gflops, "ultralytics.get_flops", ""
    except Exception as exc:
        get_flops_error = str(exc)
    else:
        get_flops_error = "ultralytics.get_flops returned zero"

    try:
        import torch
        from thop import profile as thop_profile

        param = next(model.parameters())
        dummy = torch.zeros(1, 3, img_size, img_size, device=param.device)
        flops = thop_profile(model, inputs=(dummy,), verbose=False)[0] / 1e9 * 2
        return as_float(flops), "thop.profile", ""
    except Exception as exc:
        return None, "unavailable", f"{get_flops_error}; thop fallback failed: {exc}"


def profile_item(item: Dict[str, Any], img_size: int, device: str) -> Dict[str, Any]:
    from ultralytics import YOLO

    row = dict(item)
    row.update(
        {
            "img_size": img_size,
            "device": device,
            "layers": None,
            "total_params": None,
            "trainable_params": None,
            "params_m": None,
            "trainable_params_m": None,
            "gflops": None,
            "gflops_method": "",
            "status": "ok",
            "error": "",
        }
    )
    try:
        yolo = YOLO(item["source"])
        torch_model = yolo.model
        actual_device = move_to_device(torch_model, device)

        total_params = sum(p.numel() for p in torch_model.parameters())
        trainable_params = sum(p.numel() for p in torch_model.parameters() if p.requires_grad)
        gflops, gflops_method, gflops_error = compute_gflops(torch_model, img_size)

        row.update(
            {
                "device": actual_device,
                "layers": count_layers(torch_model),
                "total_params": total_params,
                "trainable_params": trainable_params,
                "params_m": total_params / 1e6,
                "trainable_params_m": trainable_params / 1e6,
                "gflops": gflops,
                "gflops_method": gflops_method,
                "error": gflops_error,
            }
        )
        if gflops is None:
            row["status"] = "partial"
    except Exception as exc:
        row["status"] = "failed"
        row["error"] = str(exc)
    return row


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any, digits: int = 3) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def representative_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["study"], row["model"])].append(row)

    out: List[Dict[str, Any]] = []
    for (study, model), group in sorted(grouped.items()):
        ok_rows = [r for r in group if r["status"] in {"ok", "partial"}]
        rep = ok_rows[0] if ok_rows else group[0]
        out.append(
            {
                "study": study,
                "model": model,
                "n_runs": len(group),
                "representative_run": rep["run"],
                "source_type": rep["source_type"],
                "source": rep["source"],
                "model_config": rep.get("model_config", ""),
                "img_size": rep["img_size"],
                "layers": rep["layers"],
                "total_params": rep["total_params"],
                "trainable_params": rep["trainable_params"],
                "params_m": rep["params_m"],
                "trainable_params_m": rep["trainable_params_m"],
                "gflops": rep["gflops"],
                "gflops_method": rep["gflops_method"],
                "status": rep["status"],
                "error": rep["error"],
            }
        )
    return out


def read_metrics_csv(path: Path) -> Dict[tuple[str, str], Dict[str, Any]]:
    if not path or not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return {(row.get("study", ""), row.get("model", "")): row for row in reader}


def merge_metrics(
    complexity_rows: Sequence[Dict[str, Any]], metrics: Dict[tuple[str, str], Dict[str, Any]]
) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    for row in complexity_rows:
        metric_row = metrics.get((row["study"], row["model"]), {})
        merged_row = dict(row)
        for key, value in metric_row.items():
            if key in {"study", "model"}:
                continue
            merged_row[f"metric_{key}"] = value
        merged.append(merged_row)
    return merged


def write_markdown(path: Path, rows: Sequence[Dict[str, Any]], merged_rows: Sequence[Dict[str, Any]]) -> None:
    lines: List[str] = []
    lines.append("# Revision Model Complexity")
    lines.append("")
    lines.append("Parameters and GFLOPs are profiled from trained `best.pt` weights when available.")
    lines.append("GFLOPs follow the local Ultralytics/thop convention at the configured image size.")
    lines.append("")
    lines.append("## Complexity")
    lines.append("")
    lines.append("| Study | Model | n | Params (M) | GFLOPs | Layers | Source |")
    lines.append("|---|---|---:|---:|---:|---:|---|")
    for row in rows:
        lines.append(
            f"| {row['study']} | {row['model']} | {row['n_runs']} | "
            f"{fmt(row['params_m'], 3)} | {fmt(row['gflops'], 3)} | "
            f"{'' if row['layers'] is None else row['layers']} | {row['source_type']} |"
        )

    if merged_rows and any(k.startswith("metric_") for row in merged_rows for k in row):
        lines.append("")
        lines.append("## Complexity With Metrics")
        lines.append("")
        lines.append("| Study | Model | Params (M) | GFLOPs | mAP@0.5 | mAP@0.5:0.95 | Recall |")
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        for row in merged_rows:
            lines.append(
                f"| {row['study']} | {row['model']} | {fmt(row['params_m'], 3)} | "
                f"{fmt(row['gflops'], 3)} | {row.get('metric_map50_mean_std', '')} | "
                f"{row.get('metric_map50_95_mean_std', '')} | {row.get('metric_recall_mean_std', '')} |"
            )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_dry_run(items: Sequence[Dict[str, Any]]) -> None:
    printable = [
        {
            "run": item["run"],
            "study": item["study"],
            "model": item["model"],
            "seed": item["seed"],
            "source_type": item["source_type"],
            "source": item["source"],
            "model_config": item.get("model_config", ""),
        }
        for item in items
    ]
    print(json.dumps(printable, indent=2, ensure_ascii=False))


def select_representatives(items: Sequence[Dict[str, Any]], dedupe: bool) -> List[Dict[str, Any]]:
    if not dedupe:
        return list(items)
    grouped: Dict[tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for item in items:
        grouped[(item["study"], item["model"])].append(item)
    return [sorted(group, key=lambda x: (x["seed"] == "", x["seed"], x["run"]))[0] for group in grouped.values()]


def main() -> None:
    args = parse_args()
    items: List[Dict[str, Any]] = []
    if args.runs_root:
        items.extend(discover_run_items(args.runs_root))
    if args.preset:
        items.extend(preset_items(args.preset))
    if args.model:
        items.extend(explicit_model_items(args.model))

    if not items:
        raise SystemExit("No profiling inputs. Provide --runs-root, --preset, or --model NAME=PATH.")
    check_expected_count(items, args.expected_count)

    if args.dry_run:
        print_dry_run(items)
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    profile_inputs = select_representatives(items, dedupe=not args.no_dedupe)
    profile_by_key: Dict[tuple[str, str], Dict[str, Any]] = {}
    for idx, item in enumerate(profile_inputs, start=1):
        print(f"[{idx}/{len(profile_inputs)}] profiling {item['run']} from {item['source']}")
        profile_by_key[(item["study"], item["model"])] = profile_item(item, args.img_size, args.device)

    per_run_rows: List[Dict[str, Any]] = []
    for item in items:
        profile_row = profile_by_key.get((item["study"], item["model"])) or profile_item(item, args.img_size, args.device)
        row = dict(profile_row)
        row.update(
            {
                "run": item["run"],
                "seed": item["seed"],
                "source_type": item["source_type"],
                "source": item["source"],
                "weights": item["weights"],
                "args_yaml": item["args_yaml"],
                "train_imgsz": item["train_imgsz"],
                "train_data": item["train_data"],
                "profile_run": profile_row["run"],
                "profile_source": profile_row["source"],
            }
        )
        per_run_rows.append(row)

    per_model_rows = representative_rows(per_run_rows)
    metrics = read_metrics_csv(args.metrics_csv) if args.metrics_csv else {}
    merged_rows = merge_metrics(per_model_rows, metrics) if metrics else []

    write_csv(args.output_dir / "per_run_complexity.csv", per_run_rows)
    write_csv(args.output_dir / "per_model_complexity.csv", per_model_rows)
    if merged_rows:
        write_csv(args.output_dir / "complexity_with_metrics.csv", merged_rows)
    write_markdown(args.output_dir / "complexity_summary.md", per_model_rows, merged_rows)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(REPO_ROOT),
        "runs_root": str(args.runs_root) if args.runs_root else "",
        "preset": args.preset,
        "expected_count": args.expected_count,
        "img_size": args.img_size,
        "device": args.device,
        "dedupe": not args.no_dedupe,
        "metrics_csv": str(args.metrics_csv) if args.metrics_csv else "",
        "inputs": items,
        "outputs": {
            "per_run_complexity": str(args.output_dir / "per_run_complexity.csv"),
            "per_model_complexity": str(args.output_dir / "per_model_complexity.csv"),
            "complexity_with_metrics": str(args.output_dir / "complexity_with_metrics.csv")
            if merged_rows
            else "",
            "complexity_summary": str(args.output_dir / "complexity_summary.md"),
        },
    }
    (args.output_dir / "complexity_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote model complexity exports to {args.output_dir}")


if __name__ == "__main__":
    main()
