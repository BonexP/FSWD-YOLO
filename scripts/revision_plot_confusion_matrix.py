#!/usr/bin/env python3
"""Validate one detector and redraw a publication-ready confusion matrix.

The default Ultralytics plot may fail to render Chinese labels when the runtime
font fallback is incomplete. This helper reuses Ultralytics validation to obtain
the confusion matrix, then redraws it with an explicitly selected font and
exports PNG/PDF plus CSV/JSON matrix data.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export and redraw a YOLO confusion matrix.")
    parser.add_argument("--weights", required=True, help="Path to weights/best.pt.")
    parser.add_argument("--data", required=True, help="Dataset YAML.")
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
        help="Optional path to a Chinese-capable TTF/OTF font, e.g. /usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc.",
    )
    parser.add_argument(
        "--class-names",
        nargs="*",
        default=None,
        help="Optional labels overriding data.yaml/model names, e.g. Flash Hole/Pore Tunnel Burr.",
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


def normalize_columns(matrix: np.ndarray) -> np.ndarray:
    denom = matrix.sum(axis=0, keepdims=True) + 1e-9
    return matrix / denom


def write_csv_matrix(path: Path, matrix: np.ndarray, labels: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["predicted\\true", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *[f"{float(v):.8f}" for v in row]])


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
    from ultralytics import YOLO

    args.output_dir.mkdir(parents=True, exist_ok=True)
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
    labels = args.class_names or names_to_list(getattr(cm, "names", None))
    if not labels:
        labels = names_to_list(getattr(model, "names", None))

    if matrix.shape[0] == len(labels) + 1:
        labels = [*labels, "background"]
    elif matrix.shape[0] != len(labels):
        labels = [str(i) for i in range(matrix.shape[0])]

    if args.drop_background and matrix.shape[0] == matrix.shape[1] and labels and labels[-1].lower() == "background":
        matrix = matrix[:-1, :-1]
        labels = labels[:-1]

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
        "weights": str(args.weights),
        "data": str(args.data),
        "split": args.split,
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "device": args.device,
        "conf": args.conf,
        "iou": args.iou,
        "font": args.font,
        "labels": labels,
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
