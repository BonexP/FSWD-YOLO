#!/usr/bin/env python3
"""Recompute aggregate CSVs from per-seed per-class CSV.

This script reads a `per_seed_per_class.csv` (format produced by
`revision_export_per_class_metrics.py`) and recomputes:

- `per_model_per_class_mean_std.csv`: mean/std per (study, model, class)
- `overall_mean_std.csv`: approximated mean/std of overall metrics per (study, model)
- `per_seed_overall.csv`: run-level approximated overall metrics derived from per-class values

Limitations:
- `fitness` cannot be reconstructed and will be left empty.
- Overall precision/recall are approximated by averaging per-class precision/recall.
  map50 and map50_95 are computed as the mean of per-class APs.

Usage:
    python scripts/recompute_from_per_seed_per_class.py \
        --input revision_results/main_per_class/per_seed_per_class.csv \
        --output-dir revision_results/main_per_class

"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from statistics import mean, stdev
from pathlib import Path
from typing import Dict, List, Any, Iterable


PER_CLASS_METRICS = ("precision", "recall", "ap50", "ap50_95")
OVERALL_METRICS = ("precision", "recall", "map50", "map50_95", "fitness")


def fmt_mean_std(values: Iterable[float]) -> str:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return ""
    if len(vals) == 1:
        return f"{vals[0]:.4f}"
    return f"{mean(vals):.4f} +/- {stdev(vals):.4f}"


def read_csv(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def recompute(run_rows: List[Dict[str, Any]], output_dir: Path) -> None:
    # Group by run to compute approximated per-run overall metrics
    runs: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in run_rows:
        runs[r["run"]].append(r)

    per_seed_overall: List[Dict[str, Any]] = []
    for run, rows in runs.items():
        # take metadata from first row
        first = rows[0]
        study = first.get("study", "")
        model = first.get("model", "")
        seed = first.get("seed", "")

        precisions = [float(x["precision"]) for x in rows if x.get("precision") not in (None, "")]
        recalls = [float(x["recall"]) for x in rows if x.get("recall") not in (None, "")]
        ap50s = [float(x["ap50"]) for x in rows if x.get("ap50") not in (None, "")]
        ap5095s = [float(x["ap50_95"]) for x in rows if x.get("ap50_95") not in (None, "")]

        overall = {
            "run": run,
            "study": study,
            "model": model,
            "seed": seed,
            "precision": mean(precisions) if precisions else None,
            "recall": mean(recalls) if recalls else None,
            "map50": mean(ap50s) if ap50s else None,
            "map50_95": mean(ap5095s) if ap5095s else None,
            "fitness": None,
        }
        per_seed_overall.append(overall)

    # write per_seed_overall.csv
    # Use field order similar to original minimal format
    per_seed_rows_out: List[Dict[str, Any]] = []
    for r in per_seed_overall:
        per_seed_rows_out.append(
            {
                "run": r["run"],
                "study": r["study"],
                "model": r["model"],
                "seed": r["seed"],
                "precision": "" if r["precision"] is None else f"{r['precision']:.6f}",
                "recall": "" if r["recall"] is None else f"{r['recall']:.6f}",
                "map50": "" if r["map50"] is None else f"{r['map50']:.6f}",
                "map50_95": "" if r["map50_95"] is None else f"{r['map50_95']:.6f}",
                "fitness": "",
            }
        )
    write_csv(output_dir / "per_seed_overall.csv", per_seed_rows_out)

    # Aggregate overall mean/std per (study, model)
    grouped_overall: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for r in per_seed_overall:
        grouped_overall[(r["study"], r["model"])].append(r)

    overall_mean_rows: List[Dict[str, Any]] = []
    for (study, model), group in sorted(grouped_overall.items()):
        vals_prec = [g["precision"] for g in group if g["precision"] is not None]
        vals_rec = [g["recall"] for g in group if g["recall"] is not None]
        vals_map50 = [g["map50"] for g in group if g["map50"] is not None]
        vals_map5095 = [g["map50_95"] for g in group if g["map50_95"] is not None]
        row = {
            "study": study,
            "model": model,
            "n": len(group),
            "precision_mean": mean(vals_prec) if vals_prec else None,
            "precision_std": stdev(vals_prec) if len(vals_prec) > 1 else None,
            "precision_mean_std": fmt_mean_std(vals_prec) if vals_prec else "",
            "recall_mean": mean(vals_rec) if vals_rec else None,
            "recall_std": stdev(vals_rec) if len(vals_rec) > 1 else None,
            "recall_mean_std": fmt_mean_std(vals_rec) if vals_rec else "",
            "map50_mean": mean(vals_map50) if vals_map50 else None,
            "map50_std": stdev(vals_map50) if len(vals_map50) > 1 else None,
            "map50_mean_std": fmt_mean_std(vals_map50) if vals_map50 else "",
            "map50_95_mean": mean(vals_map5095) if vals_map5095 else None,
            "map50_95_std": stdev(vals_map5095) if len(vals_map5095) > 1 else None,
            "map50_95_mean_std": fmt_mean_std(vals_map5095) if vals_map5095 else "",
            "fitness_mean": None,
            "fitness_std": None,
            "fitness_mean_std": "",
        }
        overall_mean_rows.append(row)
    write_csv(output_dir / "overall_mean_std.csv", overall_mean_rows)

    # Aggregate per-class mean/std per (study, model, class)
    grouped: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for r in run_rows:
        key = (r.get("study", ""), r.get("model", ""), int(r.get("class_id", "0")), r.get("class_name", ""))
        grouped[key].append(r)

    per_class_mean_rows: List[Dict[str, Any]] = []
    for (study, model, class_id, class_name), group in sorted(grouped.items()):
        target_counts = [int(g["target_count"]) for g in group if g.get("target_count") not in (None, "")]
        stable_target_count = target_counts[0] if target_counts and len(set(target_counts)) == 1 else None
        precs = [float(g["precision"]) for g in group if g.get("precision") not in (None, "")]
        recs = [float(g["recall"]) for g in group if g.get("recall") not in (None, "")]
        ap50s = [float(g["ap50"]) for g in group if g.get("ap50") not in (None, "")]
        ap5095s = [float(g["ap50_95"]) for g in group if g.get("ap50_95") not in (None, "")]

        row = {
            "study": study,
            "model": model,
            "class_id": class_id,
            "class_name": class_name,
            "n": len(group),
            "target_count": stable_target_count if stable_target_count is not None else "",
            "precision_mean": mean(precs) if precs else None,
            "precision_std": stdev(precs) if len(precs) > 1 else None,
            "precision_mean_std": fmt_mean_std(precs) if precs else "",
            "recall_mean": mean(recs) if recs else None,
            "recall_std": stdev(recs) if len(recs) > 1 else None,
            "recall_mean_std": fmt_mean_std(recs) if recs else "",
            "ap50_mean": mean(ap50s) if ap50s else None,
            "ap50_std": stdev(ap50s) if len(ap50s) > 1 else None,
            "ap50_mean_std": fmt_mean_std(ap50s) if ap50s else "",
            "ap50_95_mean": mean(ap5095s) if ap5095s else None,
            "ap50_95_std": stdev(ap5095s) if len(ap5095s) > 1 else None,
            "ap50_95_mean_std": fmt_mean_std(ap5095s) if ap5095s else "",
        }
        per_class_mean_rows.append(row)

    write_csv(output_dir / "per_model_per_class_mean_std.csv", per_class_mean_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute aggregates from per-seed per-class CSV")
    parser.add_argument("--input", type=Path, required=True, help="path to per_seed_per_class.csv")
    parser.add_argument("--output-dir", type=Path, required=True, help="directory to write recomputed CSVs")
    args = parser.parse_args()

    rows = read_csv(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    recompute(rows, args.output_dir)


if __name__ == "__main__":
    main()
