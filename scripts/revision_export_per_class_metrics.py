#!/usr/bin/env python3
"""Re-validate revision runs and export reproducible per-class metrics."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


OVERALL_METRICS = ("precision", "recall", "map50", "map50_95", "fitness")
PER_CLASS_METRICS = ("precision", "recall", "ap50", "ap50_95")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Re-validate revision best.pt weights on a fixed split and export "
            "overall/per-class CSV, JSON, and Markdown tables."
        )
    )
    parser.add_argument("--runs-root", type=Path, default=Path("runs/revision_70_15_15_main"))
    parser.add_argument(
        "--data",
        default="/home/user/PROJECT/FSWD/FSW-MERGE_revision_dataset_augmented_double/data.yaml",
    )
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", type=Path, default=Path("revision_results/main_per_class"))
    parser.add_argument("--expected-count", type=int, default=15)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument(
        "--val-project",
        type=Path,
        default=None,
        help="Validation output directory. Defaults to <output-dir>/val_runs.",
    )
    parser.add_argument("--no-plots", action="store_true", help="Disable validation plot generation.")
    parser.add_argument("--dry-run", action="store_true", help="List discovered weights without validation.")
    return parser.parse_args()


def as_float(value: Any) -> float | None:
    if callable(value):
        value = value()
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def array_or_empty(value: Any) -> np.ndarray:
    if value is None:
        return np.array([])
    try:
        return np.asarray(value)
    except Exception:
        return np.array([])


def discover_weights(runs_root: Path) -> List[Path]:
    return sorted(runs_root.glob("*/weights/best.pt"), key=lambda p: p.parent.parent.name)


def parse_run_name(run_name: str) -> Dict[str, str]:
    match = re.fullmatch(r"(?P<study>main|ablation|abl)_(?P<model>.+)_seed(?P<seed>[0-9]+)", run_name)
    if not match:
        return {"study": "unknown", "model": run_name, "seed": ""}
    meta = match.groupdict()
    if meta["study"] == "abl":
        meta["study"] = "ablation"
    return meta


def names_to_dict(names: Any) -> Dict[int, str]:
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, (list, tuple)):
        return {i: str(v) for i, v in enumerate(names)}
    return {}


def extract_metrics(metrics: Any, model: Any) -> Dict[str, Any]:
    names = (
        getattr(metrics, "names", None)
        or getattr(getattr(model, "model", None), "names", None)
        or getattr(model, "names", None)
        or {}
    )
    names_dict = names_to_dict(names)

    box = getattr(metrics, "box", metrics)
    results_dict = getattr(metrics, "results_dict", {}) or {}
    overall = {
        "precision": as_float(getattr(box, "mp", None) or results_dict.get("metrics/precision(B)")),
        "recall": as_float(getattr(box, "mr", None) or results_dict.get("metrics/recall(B)")),
        "map50": as_float(getattr(box, "map50", None) or results_dict.get("metrics/mAP50(B)")),
        "map50_95": as_float(getattr(box, "map", None) or results_dict.get("metrics/mAP50-95(B)")),
        "fitness": as_float(getattr(metrics, "fitness", None) or results_dict.get("fitness")),
    }

    p = array_or_empty(getattr(box, "p", None))
    r = array_or_empty(getattr(box, "r", None))
    all_ap = array_or_empty(getattr(box, "all_ap", None))
    ap_class_index = array_or_empty(getattr(box, "ap_class_index", None)).astype(int)
    target_counts = array_or_empty(getattr(metrics, "nt_per_class", None))

    class_ids = sorted(names_dict)
    if not class_ids and ap_class_index.size:
        class_ids = sorted(int(i) for i in ap_class_index)

    per_class: List[Dict[str, Any]] = []
    for class_id in class_ids:
        positions = np.where(ap_class_index == class_id)[0] if ap_class_index.size else np.array([])
        pos = int(positions[0]) if positions.size else None
        ap50 = None
        ap5095 = None
        if pos is not None and all_ap.ndim == 2 and pos < all_ap.shape[0]:
            ap50 = as_float(all_ap[pos, 0])
            ap5095 = as_float(all_ap[pos].mean())
        row = {
            "class_id": class_id,
            "class_name": names_dict.get(class_id, str(class_id)),
            "target_count": int(target_counts[class_id]) if class_id < target_counts.size else None,
            "precision": as_float(p[pos]) if pos is not None and pos < p.size else None,
            "recall": as_float(r[pos]) if pos is not None and pos < r.size else None,
            "ap50": ap50,
            "ap50_95": ap5095,
        }
        per_class.append(row)

    return {"overall": overall, "per_class": per_class, "results_dict": results_dict}


def validate_weight(weight: Path, args: argparse.Namespace, val_project: Path) -> Dict[str, Any]:
    from ultralytics import YOLO

    run_name = weight.parent.parent.name
    model = YOLO(str(weight))
    metrics = model.val(
        data=str(args.data),
        split=args.split,
        imgsz=args.img_size,
        batch=args.batch_size,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        project=str(val_project),
        name=f"val_{run_name}",
        exist_ok=True,
        plots=not args.no_plots,
        save_json=False,
    )
    payload = extract_metrics(metrics, model)
    payload.update(
        {
            "run": run_name,
            "weights": str(weight),
            "data": str(args.data),
            "split": args.split,
            "save_dir": str(val_project / f"val_{run_name}"),
        }
    )
    return payload


def fmt_value(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def fmt_mean_std(values: Iterable[float | None]) -> str:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return ""
    if len(vals) == 1:
        return f"{vals[0]:.4f}"
    return f"{mean(vals):.4f} +/- {stdev(vals):.4f}"


def mean_std_fields(rows: Sequence[Dict[str, Any]], metrics: Sequence[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for metric in metrics:
        vals = [float(row[metric]) for row in rows if row.get(metric) is not None]
        out[f"{metric}_mean"] = mean(vals) if vals else None
        out[f"{metric}_std"] = stdev(vals) if len(vals) > 1 else None
        out[f"{metric}_mean_std"] = fmt_mean_std(vals)
    return out


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_rows(payloads: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    overall_rows: List[Dict[str, Any]] = []
    per_class_rows: List[Dict[str, Any]] = []
    for payload in payloads:
        meta = parse_run_name(payload["run"])
        base = {
            "run": payload["run"],
            "study": meta["study"],
            "model": meta["model"],
            "seed": meta["seed"],
            "weights": payload["weights"],
            "data": payload["data"],
            "split": payload["split"],
            "save_dir": payload["save_dir"],
        }
        overall = payload["overall"]
        overall_rows.append({**base, **{metric: overall.get(metric) for metric in OVERALL_METRICS}})
        for row in payload["per_class"]:
            per_class_rows.append(
                {
                    **base,
                    "class_id": row.get("class_id"),
                    "class_name": row.get("class_name"),
                    "target_count": row.get("target_count"),
                    **{metric: row.get(metric) for metric in PER_CLASS_METRICS},
                }
            )
    return overall_rows, per_class_rows


def aggregate_overall(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["study"], row["model"])].append(row)

    output: List[Dict[str, Any]] = []
    for (study, model), group in sorted(grouped.items()):
        output.append(
            {
                "study": study,
                "model": model,
                "n": len(group),
                **mean_std_fields(group, OVERALL_METRICS),
            }
        )
    return output


def aggregate_per_class(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple[str, str, int, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["study"], row["model"], int(row["class_id"]), row["class_name"])].append(row)

    output: List[Dict[str, Any]] = []
    for (study, model, class_id, class_name), group in sorted(grouped.items()):
        target_counts = [row.get("target_count") for row in group if row.get("target_count") is not None]
        stable_target_count = target_counts[0] if target_counts and len(set(target_counts)) == 1 else None
        output.append(
            {
                "study": study,
                "model": model,
                "class_id": class_id,
                "class_name": class_name,
                "n": len(group),
                "target_count": stable_target_count,
                **mean_std_fields(group, PER_CLASS_METRICS),
            }
        )
    return output


def write_markdown(path: Path, overall_rows: List[Dict[str, Any]], per_class_rows: List[Dict[str, Any]]) -> None:
    lines: List[str] = []
    lines.append("# Revision Per-Class Metric Tables")
    lines.append("")
    lines.append("Metrics are re-validated from `weights/best.pt` on the configured fixed split.")
    lines.append("")
    lines.append("## Overall Mean +/- Std")
    lines.append("")
    lines.append("| Study | Model | n | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | Fitness |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for row in overall_rows:
        lines.append(
            f"| {row['study']} | {row['model']} | {row['n']} | "
            f"{row['precision_mean_std']} | {row['recall_mean_std']} | "
            f"{row['map50_mean_std']} | {row['map50_95_mean_std']} | {row['fitness_mean_std']} |"
        )

    lines.append("")
    lines.append("## Per-Class Mean +/- Std")
    lines.append("")
    lines.append("| Study | Model | Class | Targets | n | Precision | Recall | AP@0.5 | AP@0.5:0.95 |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for row in per_class_rows:
        lines.append(
            f"| {row['study']} | {row['model']} | {row['class_name']} | "
            f"{'' if row['target_count'] is None else row['target_count']} | {row['n']} | "
            f"{row['precision_mean_std']} | {row['recall_mean_std']} | "
            f"{row['ap50_mean_std']} | {row['ap50_95_mean_std']} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_dry_run(weights: List[Path]) -> None:
    rows = []
    for weight in weights:
        run_name = weight.parent.parent.name
        meta = parse_run_name(run_name)
        rows.append({"run": run_name, **meta, "weights": str(weight)})
    print(json.dumps(rows, indent=2, ensure_ascii=False))


def check_expected_count(weights: List[Path], expected_count: int) -> None:
    if expected_count <= 0:
        return
    if len(weights) != expected_count:
        raise SystemExit(f"Expected {expected_count} best.pt files, found {len(weights)}.")


def main() -> None:
    args = parse_args()
    weights = discover_weights(args.runs_root)
    check_expected_count(weights, args.expected_count)

    if args.dry_run:
        print_dry_run(weights)
        return

    if not weights:
        raise SystemExit(f"No best.pt files found under {args.runs_root}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    val_project = args.val_project or (args.output_dir / "val_runs")
    val_project.mkdir(parents=True, exist_ok=True)

    payloads = []
    for idx, weight in enumerate(weights, start=1):
        print(f"[{idx}/{len(weights)}] validating {weight}")
        payloads.append(validate_weight(weight, args, val_project))

    overall_rows, per_class_rows = build_rows(payloads)
    overall_mean_rows = aggregate_overall(overall_rows)
    per_class_mean_rows = aggregate_per_class(per_class_rows)

    write_csv(args.output_dir / "per_seed_overall.csv", overall_rows)
    write_csv(args.output_dir / "per_seed_per_class.csv", per_class_rows)
    write_csv(args.output_dir / "overall_mean_std.csv", overall_mean_rows)
    write_csv(args.output_dir / "per_model_per_class_mean_std.csv", per_class_mean_rows)
    write_markdown(args.output_dir / "summary_tables.md", overall_mean_rows, per_class_mean_rows)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(REPO_ROOT),
        "runs_root": str(args.runs_root),
        "data": str(args.data),
        "split": args.split,
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "device": args.device,
        "conf": args.conf,
        "iou": args.iou,
        "expected_count": args.expected_count,
        "weights": [str(weight) for weight in weights],
        "outputs": {
            "per_seed_overall": str(args.output_dir / "per_seed_overall.csv"),
            "per_seed_per_class": str(args.output_dir / "per_seed_per_class.csv"),
            "overall_mean_std": str(args.output_dir / "overall_mean_std.csv"),
            "per_model_per_class_mean_std": str(args.output_dir / "per_model_per_class_mean_std.csv"),
            "summary_tables": str(args.output_dir / "summary_tables.md"),
        },
        "overall_rows": overall_rows,
        "per_class_rows": per_class_rows,
    }
    (args.output_dir / "export_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote per-class metric exports to {args.output_dir}")


if __name__ == "__main__":
    main()
