#!/usr/bin/env python3
"""Validate one detector and redraw a publication-ready confusion matrix.

By default, labels are converted to the English class names used in the paper:
Flash, Hole/Pore, Tunnel, and Burr. The script can also redraw a manually edited
matrix from CSV or apply explicit cell overrides to a validated matrix. Manual
changes are recorded in the manifest for traceability.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


PAPER_CLASS_NAMES = ["Flash", "Hole/Pore", "Tunnel", "Burr"]
ENGLISH_NAME_MAP = {
    "飞边": "Flash",
    "flash": "Flash",
    "孔洞": "Hole/Pore",
    "孔洞/气孔": "Hole/Pore",
    "hole": "Hole/Pore",
    "hole/pore": "Hole/Pore",
    "pore": "Hole/Pore",
    "隧道": "Tunnel",
    "tunnel": "Tunnel",
    "毛刺": "Burr",
    "burr": "Burr",
    "background": "background",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export and redraw a YOLO confusion matrix.")
    parser.add_argument("--weights", default=None, help="Path to weights/best.pt.")
    parser.add_argument("--data", default=None, help="Dataset YAML.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", default="confusion_matrix")
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument(
        "--font",
        default=None,
        help="Optional path to a TTF/OTF/TTC font file.",
    )
    parser.add_argument(
        "--class-names",
        nargs="*",
        default=None,
        help="Labels overriding data.yaml/model names, e.g. Flash Hole/Pore Tunnel Burr.",
    )
    parser.add_argument(
        "--matrix-csv",
        type=Path,
        default=None,
        help=(
            "Optional manual matrix CSV to redraw without validation. The first row should be labels and "
            "the first column should be predicted labels, matching exported *_raw.csv format."
        ),
    )
    parser.add_argument(
        "--set-cell",
        action="append",
        default=[],
        metavar="PREDICTED,TRUE,VALUE",
        help=(
            "Override one matrix cell after validation/CSV loading. PREDICTED and TRUE may be zero-based "
            "indices or label names. May be repeated."
        ),
    )
    parser.add_argument(
        "--drop-background",
        action="store_true",
        help="Only plot defect classes and omit the background row/column.",
    )
    parser.add_argument("--figsize", default="8,6", help="Matplotlib figure size, width,height.")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--cmap", default="Blues")
    return parser.parse_args()


def names_to_list(names: Any) -> List[str]:
    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(names)]
    if isinstance(names, (list, tuple)):
        return [str(v) for v in names]
    return []


def to_paper_label(label: str) -> str:
    key = str(label).strip()
    return ENGLISH_NAME_MAP.get(key, ENGLISH_NAME_MAP.get(key.lower(), key))


def to_paper_labels(labels: List[str]) -> List[str]:
    mapped = [to_paper_label(label) for label in labels]
    if len(mapped) == len(PAPER_CLASS_NAMES) and all(label not in {"0", "1", "2", "3"} for label in mapped):
        return mapped
    if len(mapped) == len(PAPER_CLASS_NAMES):
        return PAPER_CLASS_NAMES.copy()
    if len(mapped) == len(PAPER_CLASS_NAMES) + 1 and mapped[-1].lower() == "background":
        return [*PAPER_CLASS_NAMES, "background"]
    return mapped


def load_font_properties(font_path: str | None) -> Any:
    from matplotlib import font_manager

    candidates = []
    if font_path:
        candidates.append(Path(font_path))
    candidates.extend(
        Path(p)
        for p in (
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.otf",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            "/usr/share/fonts/truetype/arphic/ukai.ttc",
            "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/simhei.ttf",
            "C:/Windows/Fonts/simsun.ttc",
        )
    )
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        return None
    if not path.exists():
        raise SystemExit(f"Font file does not exist: {path}")
    return font_manager.FontProperties(fname=str(path))


def read_csv_matrix(path: Path) -> Tuple[np.ndarray, List[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        raise SystemExit(f"Matrix CSV is empty: {path}")

    header = rows[0]
    labels = [cell.strip() for cell in header[1:]]
    matrix_rows = rows[1:]
    row_labels = [row[0].strip() for row in matrix_rows if row]
    values = []
    for row in matrix_rows:
        if not row:
            continue
        values.append([float(cell) for cell in row[1:]])

    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise SystemExit(f"Matrix CSV must contain a square matrix: {path}")
    if labels and len(labels) == matrix.shape[1]:
        return matrix, labels
    if row_labels and len(row_labels) == matrix.shape[0]:
        return matrix, row_labels
    return matrix, [str(i) for i in range(matrix.shape[0])]


def normalize_columns(matrix: np.ndarray) -> np.ndarray:
    denom = matrix.sum(axis=0, keepdims=True) + 1e-9
    return matrix / denom


def write_csv_matrix(path: Path, matrix: np.ndarray, labels: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["predicted\\true", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *[f"{float(v):.8f}" for v in row]])


def resolve_axis_index(token: str, labels: List[str]) -> int:
    token = token.strip()
    try:
        index = int(token)
    except ValueError:
        lowered = token.lower()
        lookup = {label.lower(): i for i, label in enumerate(labels)}
        if lowered not in lookup:
            raise SystemExit(f"Unknown matrix label in --set-cell: {token}. Known labels: {labels}")
        index = lookup[lowered]
    if index < 0 or index >= len(labels):
        raise SystemExit(f"Matrix index out of range in --set-cell: {token}")
    return index


def apply_cell_overrides(matrix: np.ndarray, labels: List[str], overrides: List[str]) -> List[Dict[str, Any]]:
    applied: List[Dict[str, Any]] = []
    for spec in overrides:
        parts = [part.strip() for part in spec.split(",")]
        if len(parts) != 3:
            raise SystemExit(f"--set-cell expects PREDICTED,TRUE,VALUE, got: {spec}")
        pred_idx = resolve_axis_index(parts[0], labels)
        true_idx = resolve_axis_index(parts[1], labels)
        value = float(parts[2])
        before = float(matrix[pred_idx, true_idx])
        matrix[pred_idx, true_idx] = value
        applied.append(
            {
                "predicted": labels[pred_idx],
                "true": labels[true_idx],
                "predicted_index": pred_idx,
                "true_index": true_idx,
                "old_value": before,
                "new_value": value,
            }
        )
    return applied


def plot_matrix(
    matrix: np.ndarray,
    labels: List[str],
    path: Path,
    *,
    normalized: bool,
    font_prop: Any,
    figsize: tuple[float, float],
    dpi: int,
    cmap: str,
) -> None:
    import matplotlib.pyplot as plt

    values = normalize_columns(matrix) if normalized else matrix.astype(float)
    annot_values = values.copy()
    if normalized:
        annot_values[annot_values < 0.005] = np.nan

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(values, cmap=cmap, vmin=0.0, interpolation="none")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ticks = np.arange(len(labels))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontproperties=font_prop)
    ax.set_yticklabels(labels, fontproperties=font_prop)
    ax.set_xlabel("True", fontproperties=font_prop)
    ax.set_ylabel("Predicted", fontproperties=font_prop)
    ax.set_title("Confusion Matrix Normalized" if normalized else "Confusion Matrix", fontproperties=font_prop)

    threshold = 0.45 * np.nanmax(values) if values.size and np.isfinite(values).any() else 0.0
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = annot_values[i, j]
            if np.isnan(val):
                continue
            text = f"{val:.2f}" if normalized else f"{int(round(val))}"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="white" if values[i, j] > threshold else "black",
                fontproperties=font_prop,
                fontsize=9,
            )

    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source = "validation"
    metrics = None
    model = None
    if args.matrix_csv:
        matrix, labels = read_csv_matrix(args.matrix_csv)
        source = "matrix_csv"
    else:
        if not args.weights or not args.data:
            raise SystemExit("--weights and --data are required unless --matrix-csv is provided.")
        from ultralytics import YOLO

        val_project = args.output_dir / "_ultralytics_val"
        model = YOLO(args.weights)
        metrics = model.val(
            data=args.data,
            split=args.split,
            imgsz=args.img_size,
            batch=args.batch_size,
            device=args.device,
            conf=args.conf,
            iou=args.iou,
            project=str(val_project),
            name=args.name,
            exist_ok=True,
            plots=True,
            save_json=False,
        )

        cm = getattr(metrics, "confusion_matrix", None)
        if cm is None:
            raise SystemExit("Validation did not return a confusion matrix.")

        matrix = np.asarray(cm.matrix, dtype=float)
        labels = names_to_list(getattr(cm, "names", None))
        if not labels:
            labels = names_to_list(getattr(model, "names", None))

    labels = args.class_names or to_paper_labels(labels)

    if matrix.shape[0] == len(labels) + 1:
        labels = [*labels, "background"]
    elif matrix.shape[0] != len(labels):
        labels = [str(i) for i in range(matrix.shape[0])]

    if args.drop_background and matrix.shape[0] == matrix.shape[1] and labels and labels[-1].lower() == "background":
        matrix = matrix[:-1, :-1]
        labels = labels[:-1]

    applied_overrides = apply_cell_overrides(matrix, labels, args.set_cell)

    width, height = (float(x.strip()) for x in args.figsize.split(",", 1))
    font_prop = load_font_properties(args.font)

    raw_csv = args.output_dir / f"{args.name}_raw.csv"
    normalized_csv = args.output_dir / f"{args.name}_normalized.csv"
    write_csv_matrix(raw_csv, matrix, labels)
    write_csv_matrix(normalized_csv, normalize_columns(matrix), labels)

    outputs = {}
    for normalized in (False, True):
        suffix = "normalized" if normalized else "raw"
        for ext in ("png", "pdf"):
            out_path = args.output_dir / f"{args.name}_{suffix}.{ext}"
            plot_matrix(
                matrix,
                labels,
                out_path,
                normalized=normalized,
                font_prop=font_prop,
                figsize=(width, height),
                dpi=args.dpi,
                cmap=args.cmap,
            )
            outputs[f"{suffix}_{ext}"] = str(out_path)

    manifest: Dict[str, Any] = {
        "source": source,
        "weights": str(args.weights) if args.weights else None,
        "data": str(args.data) if args.data else None,
        "matrix_csv": str(args.matrix_csv) if args.matrix_csv else None,
        "split": args.split,
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "device": args.device,
        "conf": args.conf,
        "iou": args.iou,
        "font": args.font,
        "labels": labels,
        "manual_cell_overrides": applied_overrides,
        "matrix": matrix.tolist(),
        "outputs": {
            "raw_csv": str(raw_csv),
            "normalized_csv": str(normalized_csv),
            **outputs,
        },
    }
    manifest_path = args.output_dir / f"{args.name}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest["outputs"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
