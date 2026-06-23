from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

import albumentations as A
import cv2
import yaml


# 性能优化：防止 OpenCV 线程竞争，在多 worker DataLoader 中尤其重要。
cv2.setNumThreads(0)

TARGET_SIZE = 640
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_DATA_YAML = "/home/user/PROJECT/FSWD/FSW-MERGE_revision_dataset/data.yaml"


# ==============================================================================
# 增强管道配置
# ==============================================================================

# 方案1：安全模式。使用 Resize 避免裁剪，尽量保留所有 bbox。
transform_safe = A.Compose([
    A.Resize(height=TARGET_SIZE, width=TARGET_SIZE, p=1.0),
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.3),
    A.Affine(
        scale=(0.9, 1.1),
        rotate=(-10, 10),
        p=0.4,
        border_mode=cv2.BORDER_CONSTANT,
        fill=114,
    ),
    A.RandomBrightnessContrast(
        brightness_limit=0.2,
        contrast_limit=0.2,
        p=0.5,
    ),
    A.HueSaturationValue(
        hue_shift_limit=10,
        sat_shift_limit=20,
        val_shift_limit=10,
        p=0.3,
    ),
    A.OneOf([
        A.GaussianBlur(blur_limit=5, p=1.0),
        A.MotionBlur(blur_limit=5, p=1.0),
    ], p=0.2),
    A.GaussNoise(std_range=(0.01, 0.05), mean_range=(0.0, 0.0), p=0.2),
], bbox_params=A.BboxParams(
    format="yolo",
    label_fields=["class_labels"],
    min_area=0,
    min_visibility=0,
))

# 方案2：激进模式。增强更强，但可能丢失少量 bbox。
transform_aggressive = A.Compose([
    A.Resize(height=int(TARGET_SIZE * 1.2), width=int(TARGET_SIZE * 1.2), p=1.0),
    A.RandomCrop(height=TARGET_SIZE, width=TARGET_SIZE, p=0.5),
    A.Resize(height=TARGET_SIZE, width=TARGET_SIZE, p=1.0),
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.3),
    A.OneOf([
        A.CoarseDropout(
            num_holes_range=(2, 4),
            hole_height_range=(8, 24),
            hole_width_range=(8, 24),
            fill=0,
            p=1.0,
        ),
        A.CoarseDropout(
            num_holes_range=(1, 1),
            hole_height_range=(0.05, 0.08),
            hole_width_range=(0.05, 0.08),
            fill=0,
            p=1.0,
        ),
    ], p=0.3),
    A.Affine(
        scale=(0.85, 1.15),
        rotate=(-12, 12),
        p=0.4,
        border_mode=cv2.BORDER_CONSTANT,
        fill=114,
    ),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
    A.OneOf([
        A.GaussianBlur(blur_limit=5, p=1.0),
        A.MotionBlur(blur_limit=5, p=1.0),
    ], p=0.2),
    A.GaussNoise(std_range=(0.01, 0.05), mean_range=(0.0, 0.0), p=0.2),
], bbox_params=A.BboxParams(
    format="yolo",
    label_fields=["class_labels"],
    min_area=16,
    min_visibility=0.3,
))


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Augment only the train subset of a YOLO dataset and copy val/test "
            "unchanged. The script supports both images/Train and images/train "
            "layouts, plus data.yaml entries that point to txt files."
        )
    )
    parser.add_argument(
        "--data-yaml",
        default=DEFAULT_DATA_YAML,
        help="Input dataset YAML. Default targets the materialized revision dataset.",
    )
    parser.add_argument(
        "--output-double",
        default=None,
        help="Output directory for the 2x dataset. Defaults to '<dataset>_augmented_double'.",
    )
    parser.add_argument(
        "--output-quadruple",
        default=None,
        help="Output directory for the 4x dataset. Defaults to '<dataset>_augmented_quadruple'.",
    )
    parser.add_argument(
        "--mode",
        choices=["safe", "aggressive"],
        default="safe",
        help="Augmentation mode. Safe mode is recommended for small weld defects.",
    )
    parser.add_argument(
        "--skip-double",
        action="store_true",
        help="Do not create the 2x augmented dataset.",
    )
    parser.add_argument(
        "--skip-quadruple",
        action="store_true",
        help="Do not create the 4x augmented dataset.",
    )
    parser.add_argument(
        "--include-existing-augmented",
        action="store_true",
        help="Also augment files whose stem contains '_aug_'. By default they are skipped.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove an existing output directory before writing a new one.",
    )
    return parser.parse_args()


def normalize_bbox(bbox):
    """Clamp YOLO bbox coordinates into [0.0, 1.0]."""
    x_center, y_center, w, h = bbox
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))

    if x_center - w / 2 < 0:
        x_center = w / 2
    if x_center + w / 2 > 1:
        x_center = 1 - w / 2
    if y_center - h / 2 < 0:
        y_center = h / 2
    if y_center + h / 2 > 1:
        y_center = 1 - h / 2

    return [x_center, y_center, w, h]


def resolve_case_insensitive(path):
    """Resolve a path while tolerating Train/train style case differences."""
    path = Path(path).expanduser()
    if path.exists():
        return path

    current = Path(path.anchor) if path.is_absolute() else Path(".")
    parts = path.parts[1:] if path.is_absolute() else path.parts
    for part in parts:
        candidate = current / part
        if candidate.exists():
            current = candidate
            continue
        if current.exists() and current.is_dir():
            matches = [child for child in current.iterdir() if child.name.lower() == part.lower()]
            if matches:
                current = matches[0]
                continue
        current = candidate
    return current


def load_data_yaml(data_yaml_path):
    data_yaml_path = resolve_case_insensitive(Path(data_yaml_path))
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml_path}")

    with data_yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    dataset_root = Path(data.get("path") or data_yaml_path.parent).expanduser()
    if not dataset_root.is_absolute():
        dataset_root = data_yaml_path.parent / dataset_root
    dataset_root = resolve_case_insensitive(dataset_root)
    return data_yaml_path, data, dataset_root


def split_entry(data, split_name):
    candidates = {
        "train": ["train", "Train", "TRAIN"],
        "val": ["val", "Val", "valid", "Valid", "validation", "Validation", "VAL"],
        "test": ["test", "Test", "TEST"],
    }[split_name]
    for key in candidates:
        if key in data:
            return data[key]
    return None


def resolve_dataset_path(dataset_root, raw_path):
    raw_path = Path(str(raw_path)).expanduser()
    if raw_path.is_absolute():
        candidates = [raw_path]
    else:
        candidates = [dataset_root / raw_path]
        if raw_path.parts and raw_path.parts[0].lower() == "data":
            candidates.append(dataset_root / Path(*raw_path.parts[1:]))

    for candidate in candidates:
        resolved = resolve_case_insensitive(candidate)
        if resolved.exists():
            return resolved
    return resolve_case_insensitive(candidates[0])


def list_images(directory):
    directory = resolve_case_insensitive(directory)
    if not directory.exists():
        return []
    return sorted(
        [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda p: p.name.lower(),
    )


def read_image_txt(txt_path, dataset_root):
    images = []
    with txt_path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            image_path = resolve_dataset_path(dataset_root, raw)
            images.append(image_path)
    return images


def default_subset_images(dataset_root, split_name):
    aliases = {
        "train": ["train", "Train"],
        "val": ["val", "Val", "valid", "validation"],
        "test": ["test", "Test"],
    }[split_name]
    for alias in aliases:
        image_dir = resolve_case_insensitive(dataset_root / "images" / alias)
        images = list_images(image_dir)
        if images:
            return images
    return []


def load_split_images(data, dataset_root, split_name):
    entry = split_entry(data, split_name)
    if entry is None:
        return default_subset_images(dataset_root, split_name)

    entries = entry if isinstance(entry, list) else [entry]
    images = []
    for item in entries:
        path = resolve_dataset_path(dataset_root, item)
        if path.is_file() and path.suffix.lower() == ".txt":
            images.extend(read_image_txt(path, dataset_root))
        elif path.is_dir():
            images.extend(list_images(path))
        elif path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(path)
        else:
            raise FileNotFoundError(f"Cannot resolve {split_name} entry: {item} -> {path}")

    seen = set()
    unique_images = []
    for image_path in images:
        key = str(image_path)
        if key not in seen:
            unique_images.append(image_path)
            seen.add(key)
    return unique_images


def infer_label_path(image_path):
    parts = list(Path(image_path).parts)
    for idx, part in enumerate(parts):
        if part.lower() == "images":
            parts[idx] = "labels"
            return Path(*parts).with_suffix(".txt")
    return Path(image_path).with_suffix(".txt")


def read_yolo_label(label_path):
    bboxes = []
    class_labels = []
    if not label_path.exists():
        return bboxes, class_labels

    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            class_id = int(float(parts[0]))
            bbox = [float(value) for value in parts[1:]]
            bboxes.append(normalize_bbox(bbox))
            class_labels.append(class_id)
    return bboxes, class_labels


def unique_image_name(image_path, used_names):
    image_path = Path(image_path)
    name = image_path.name
    if name not in used_names:
        used_names.add(name)
        return name

    digest = hashlib.sha1(str(image_path).encode("utf-8")).hexdigest()[:8]
    name = f"{image_path.stem}_{digest}{image_path.suffix}"
    used_names.add(name)
    return name


def ensure_output_dir(output_path, dataset_root, overwrite):
    output_path = Path(output_path).expanduser()
    dataset_root = Path(dataset_root).expanduser()
    if output_path.resolve() == dataset_root.resolve():
        raise ValueError(f"Output directory must not equal input dataset root: {output_path}")

    if output_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_path}. "
                "Use --overwrite or choose another output path."
            )
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def relative_image_path(split_name, image_name):
    return Path("images", split_name, image_name).as_posix()


def copy_label_or_empty(label_path, output_label_path):
    if label_path.exists():
        shutil.copy2(label_path, output_label_path)
        return False
    output_label_path.write_text("", encoding="utf-8")
    return True


def augment_dataset(image_paths, output_img_dir, output_label_dir, multiplier, transform):
    """
    Copy original train images and generate augmented versions.

    multiplier=1 creates a 2x train subset; multiplier=3 creates a 4x train subset.
    """
    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)

    print(f"找到 {len(image_paths)} 张 train 图像待增强")
    rel_images = []
    used_image_names = set()
    total_augmented = 0
    failed_count = 0
    missing_labels = 0
    bbox_loss_count = 0
    partial_bbox_loss_count = 0
    total_original_bboxes = 0
    total_retained_bboxes = 0

    for idx, image_path in enumerate(image_paths):
        if (idx + 1) % 100 == 0:
            if total_original_bboxes > 0:
                retention_rate = (total_retained_bboxes / total_original_bboxes) * 100
                print(
                    f"处理进度: {idx + 1}/{len(image_paths)} | "
                    f"Bbox保留率: {retention_rate:.1f}% | "
                    f"完全丢失: {bbox_loss_count} | 部分丢失: {partial_bbox_loss_count}"
                )
            else:
                print(f"处理进度: {idx + 1}/{len(image_paths)}")

        image_path = Path(image_path)
        label_path = infer_label_path(image_path)
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"警告：无法读取图像 {image_path}")
            failed_count += 1
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bboxes, class_labels = read_yolo_label(label_path)

        copied_image_name = unique_image_name(image_path, used_image_names)
        copied_label_name = Path(copied_image_name).with_suffix(".txt").name
        shutil.copy2(image_path, output_img_dir / copied_image_name)
        if copy_label_or_empty(label_path, output_label_dir / copied_label_name):
            missing_labels += 1
        rel_images.append(relative_image_path("train", copied_image_name))

        base_name = Path(copied_image_name).stem
        for i in range(multiplier):
            max_retries = 5
            for retry in range(max_retries):
                try:
                    transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
                    transformed_image = transformed["image"]
                    transformed_bboxes = transformed["bboxes"]
                    transformed_class_labels = transformed["class_labels"]

                    original_bbox_count = len(bboxes)
                    retained_bbox_count = len(transformed_bboxes)
                    if original_bbox_count > 0 and retained_bbox_count == 0:
                        if retry < max_retries - 1:
                            continue
                        print(
                            f"  警告: {image_path.name} 增强 {i + 1}: "
                            f"所有 {original_bbox_count} 个 bbox 丢失，跳过此增强"
                        )
                        bbox_loss_count += 1
                        failed_count += 1
                        break
                    if original_bbox_count > 0 and retained_bbox_count < original_bbox_count:
                        partial_bbox_loss_count += 1
                        if retained_bbox_count < original_bbox_count * 0.5:
                            print(
                                f"  警告: {image_path.name} 增强 {i + 1}: "
                                f"bbox 从 {original_bbox_count} 减少到 {retained_bbox_count}"
                            )

                    total_original_bboxes += original_bbox_count
                    total_retained_bboxes += retained_bbox_count

                    aug_image_name = f"{base_name}_aug_{i}.jpg"
                    if aug_image_name in used_image_names:
                        digest = hashlib.sha1(f"{image_path}:{i}".encode("utf-8")).hexdigest()[:8]
                        aug_image_name = f"{base_name}_aug_{i}_{digest}.jpg"
                    used_image_names.add(aug_image_name)

                    cv2.imwrite(
                        str(output_img_dir / aug_image_name),
                        cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR),
                    )
                    with (output_label_dir / Path(aug_image_name).with_suffix(".txt").name).open(
                        "w", encoding="utf-8"
                    ) as f:
                        for bbox, class_id in zip(transformed_bboxes, transformed_class_labels):
                            f.write(f"{int(class_id)} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")

                    rel_images.append(relative_image_path("train", aug_image_name))
                    total_augmented += 1
                    break
                except Exception as exc:
                    if retry < max_retries - 1:
                        continue
                    print(f"  增强失败 {image_path.name} 尝试 {i + 1}: {exc}")
                    failed_count += 1
                    break

    print_summary(
        split_name="train",
        original_count=len(image_paths),
        total_augmented=total_augmented,
        failed_count=failed_count,
        missing_labels=missing_labels,
        bbox_loss_count=bbox_loss_count,
        partial_bbox_loss_count=partial_bbox_loss_count,
        total_original_bboxes=total_original_bboxes,
        total_retained_bboxes=total_retained_bboxes,
        multiplier=multiplier,
    )
    return rel_images


def copy_subset(image_paths, split_name, output_img_dir, output_label_dir):
    """Copy val/test subsets without augmentation."""
    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)

    rel_images = []
    used_image_names = set()
    missing_labels = 0
    for image_path in image_paths:
        image_path = Path(image_path)
        copied_image_name = unique_image_name(image_path, used_image_names)
        copied_label_name = Path(copied_image_name).with_suffix(".txt").name
        shutil.copy2(image_path, output_img_dir / copied_image_name)
        if copy_label_or_empty(infer_label_path(image_path), output_label_dir / copied_label_name):
            missing_labels += 1
        rel_images.append(relative_image_path(split_name, copied_image_name))

    print(f"{split_name} 子集复制完成: {len(rel_images)} 图像, 缺失标签 {missing_labels}")
    return rel_images


def print_summary(
    split_name,
    original_count,
    total_augmented,
    failed_count,
    missing_labels,
    bbox_loss_count,
    partial_bbox_loss_count,
    total_original_bboxes,
    total_retained_bboxes,
    multiplier,
):
    print(f"\n{'=' * 70}")
    print(f"{split_name} 增强统计报告")
    print(f"{'=' * 70}")
    print(f"  - 原始图像: {original_count}")
    print(f"  - 成功增强: {total_augmented}")
    print(f"  - 失败次数: {failed_count}")
    print(f"  - 缺失标签: {missing_labels}")
    print(f"  - 总图像数: {original_count + total_augmented}")
    print("Bbox 保留统计:")
    print(f"  - 原始 bbox 总数: {total_original_bboxes}")
    print(f"  - 保留 bbox 总数: {total_retained_bboxes}")
    if total_original_bboxes > 0:
        retention_rate = (total_retained_bboxes / total_original_bboxes) * 100
        loss_rate = (bbox_loss_count / (original_count * multiplier)) * 100 if multiplier > 0 else 0
        partial_loss_rate = (
            (partial_bbox_loss_count / (original_count * multiplier)) * 100 if multiplier > 0 else 0
        )
        print(f"  - 总体保留率: {retention_rate:.2f}%")
        print(f"  - 完全丢失 bbox 的增强: {bbox_loss_count} ({loss_rate:.2f}%)")
        print(f"  - 部分丢失 bbox 的增强: {partial_bbox_loss_count} ({partial_loss_rate:.2f}%)")
        if retention_rate < 90:
            print(f"警告: Bbox 保留率较低 ({retention_rate:.1f}%)，建议使用 safe 模式。")
    print(f"{'=' * 70}\n")


def write_split_txt(output_path, split_name, rel_images):
    txt_path = output_path / f"{split_name}.txt"
    txt_path.write_text("\n".join(rel_images) + ("\n" if rel_images else ""), encoding="utf-8")


def write_data_yaml(original_data, output_path, has_test):
    data = dict(original_data)
    for key in ["Train", "Val", "Valid", "Validation", "Test", "TRAIN", "VAL", "TEST"]:
        data.pop(key, None)
    data["path"] = str(output_path)
    data["train"] = "train.txt"
    data["val"] = "val.txt"
    if has_test:
        data["test"] = "test.txt"
    else:
        data.pop("test", None)

    with (output_path / "data.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
    print(f"YAML 配置已更新: {output_path / 'data.yaml'}")


def run_variant(name, multiplier, output_path, data, dataset_root, train_images, val_images, test_images, transform, overwrite):
    output_path = ensure_output_dir(output_path, dataset_root, overwrite)
    print("=" * 70)
    print(f"正在创建 {name} 数据集: {output_path}")
    print("=" * 70)

    train_rel = augment_dataset(
        train_images,
        output_path / "images" / "train",
        output_path / "labels" / "train",
        multiplier=multiplier,
        transform=transform,
    )
    val_rel = copy_subset(
        val_images,
        "val",
        output_path / "images" / "val",
        output_path / "labels" / "val",
    )
    test_rel = []
    if test_images:
        test_rel = copy_subset(
            test_images,
            "test",
            output_path / "images" / "test",
            output_path / "labels" / "test",
        )

    write_split_txt(output_path, "train", train_rel)
    write_split_txt(output_path, "val", val_rel)
    if test_images:
        write_split_txt(output_path, "test", test_rel)
    write_data_yaml(data, output_path, has_test=bool(test_images))
    print(f"{name} 数据集完成: {output_path}\n")


def main():
    args = parse_args()
    data_yaml_path, data, dataset_root = load_data_yaml(args.data_yaml)

    train_images = load_split_images(data, dataset_root, "train")
    val_images = load_split_images(data, dataset_root, "val")
    test_images = load_split_images(data, dataset_root, "test")

    if not args.include_existing_augmented:
        before = len(train_images)
        train_images = [p for p in train_images if "_aug_" not in p.stem]
        filtered = before - len(train_images)
    else:
        filtered = 0

    if not train_images:
        raise RuntimeError("No train images found. Check data.yaml and images/train or images/Train.")
    if not val_images:
        raise RuntimeError("No val images found. The revision workflow requires a validation subset.")

    transform = transform_safe if args.mode == "safe" else transform_aggressive
    output_double = Path(args.output_double).expanduser() if args.output_double else dataset_root.with_name(
        f"{dataset_root.name}_augmented_double"
    )
    output_quadruple = (
        Path(args.output_quadruple).expanduser()
        if args.output_quadruple
        else dataset_root.with_name(f"{dataset_root.name}_augmented_quadruple")
    )

    print(f"\n{'=' * 70}")
    print(f"输入 YAML: {data_yaml_path}")
    print(f"输入数据集根目录: {dataset_root}")
    print(f"增强模式: {args.mode}")
    print(f"train/val/test 图像数: {len(train_images)}/{len(val_images)}/{len(test_images)}")
    print(f"已过滤已有增强样本: {filtered}")
    print("输出子集命名: images/train, images/val, images/test")
    print(f"{'=' * 70}\n")

    if not args.skip_double:
        run_variant(
            "双倍增强",
            multiplier=1,
            output_path=output_double,
            data=data,
            dataset_root=dataset_root,
            train_images=train_images,
            val_images=val_images,
            test_images=test_images,
            transform=transform,
            overwrite=args.overwrite,
        )

    if not args.skip_quadruple:
        run_variant(
            "四倍增强",
            multiplier=3,
            output_path=output_quadruple,
            data=data,
            dataset_root=dataset_root,
            train_images=train_images,
            val_images=val_images,
            test_images=test_images,
            transform=transform,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
