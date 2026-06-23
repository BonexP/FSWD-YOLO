import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ultralytics import YOLO
import ultralytics.data.build as build
from ultralytics.data.weighted_dataset import YOLOWeightedDataset
from ultralytics.utils.trainer_utils import print_training_config


def parse_args():
    parser = argparse.ArgumentParser(description='YOLO11 Baseline Training Script')
    parser.add_argument('--cfg', type=str, default='/home/user/PROJECT/FSWD/FSW-MERGE/data.yaml',
                        help='数据集配置文件路径')
    parser.add_argument('--model', type=str, default='ultralytics/cfg/models/11/yolo11s.yaml',
                        help='模型配置文件路径')
    parser.add_argument('--epochs', type=int, default=300, help='训练 epochs')
    # 允许 -1 自动探测最大 batch
    parser.add_argument('--batch-size', type=int, default=16, help='batch sizem,默认为16 ，建议 64/128，或设为 -1 自动探测')
    parser.add_argument('--img-size', type=int, default=640, help='输入图片大小')
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['SGD', 'Adam'], help='优化器')
    parser.add_argument('--lr0', type=float, default=0.001, help='初始学习率')
    parser.add_argument('--weight-decay', type=float, default=0.0005, help='权重衰减')
    parser.add_argument('--momentum', type=float, default=0.937, help='动量')

    # 损失函数选择
    parser.add_argument('--iou-type', type=str, default='CIoU', 
                        choices=['IoU', 'GIoU', 'DIoU', 'CIoU', 'ShapeIoU'],
                        help='IoU 损失函数类型 (默认: CIoU)')

    # 性能关键参数
    parser.add_argument('--device', type=str, default='0', help='CUDA 设备，如 0 或 0,1')
    parser.add_argument('--workers', type=int, default=16, help='DataLoader workers 数')
    parser.add_argument('--cache', type=str, default='ram', choices=['', 'ram', 'disk'],
                        help='缓存数据集到内存或磁盘，加速 IO')
    parser.add_argument('--deterministic', action='store_true', default=True,
                        help='确定性训练（会变慢，默认打开） 。强制使用确定性算法，确保可重复性，但由于限制了非确定性算法，可能会影响性能和速度。')
    parser.add_argument('--rect', action='store_true', default=False,
                        help='矩形训练以减少 padding/resize 开销（默认关闭） 。启用最小填充策略——批量中的图像被最小程度地填充以达到一个共同的大小，最长边等于 imgsz。可以提高效率和速度，但可能会影响模型精度。')

    # 数据增强总开关
    parser.add_argument('--augment', action='store_true', help='启用数据增强')
    parser.add_argument('--auto-augment', type=str, default='randaugment')
    parser.add_argument('--mosaic', type=float, default=1.0)
    parser.add_argument('--mixup', type=float, default=0.2)
    parser.add_argument('--hsv-h', type=float, default=0.015)
    parser.add_argument('--hsv-s', type=float, default=0.7)
    parser.add_argument('--hsv-v', type=float, default=0.4)
    parser.add_argument('--translate', type=float, default=0.1)
    parser.add_argument('--scale', type=float, default=0.5)
    parser.add_argument('--fliplr', type=float, default=0.5)
    parser.add_argument('--erasing', type=float, default=0.4)
    parser.add_argument('--warmup-epochs', type=int, default=5)
    parser.add_argument('--close-mosaic', type=int, default=10)

    parser.add_argument('--project', type=str, default='runs/train', help='结果保存目录')
    parser.add_argument('--name', type=str, default='baseline_yolo11', help='实验名')
    parser.add_argument('--seed', type=int, default=0, help='训练随机种子')
    # 新增：是否启用加权 DataLoader
    parser.add_argument('--weighted-dataloader', action='store_true',
                        help='启用加权 YOLOWeightedDataset 以缓解类别不平衡')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    print_training_config(args)

    save_dir = Path(args.project) / args.name
    save_dir.mkdir(parents=True, exist_ok=True)

    custom_yaml = args.model
    model = YOLO(custom_yaml)
    with open(custom_yaml, 'r', encoding='utf-8') as f:
        print("YAML 文件正文如下：\n" + f.read())

    if args.augment:
        augment_config = {
            'augment': True,
            'auto_augment': args.auto_augment,
            'mosaic': args.mosaic,
            'mixup': args.mixup,
            'hsv_h': args.hsv_h,
            'hsv_s': args.hsv_s,
            'hsv_v': args.hsv_v,
            'translate': args.translate,
            'scale': args.scale,
            'fliplr': args.fliplr,
            'erasing': args.erasing,
        }
        print("✅ 数据增强已启用")
    else:
        augment_config = {
            'augment': False, 'auto_augment': None, 'mosaic': 0.0, 'mixup': 0.0,
            'hsv_h': 0.0, 'hsv_s': 0.0, 'hsv_v': 0.0, 'translate': 0.0,
            'scale': 0.0, 'fliplr': 0.0, 'erasing': 0.0,
        }
        print("❌ 数据增强已关闭")

    # 是否启用加权 DataLoader 的日志和猴子补丁
    if args.weighted_dataloader:
        build.YOLODataset = YOLOWeightedDataset
        print("✅ 已启用加权 DataLoader: ultralytics.data.weighted_dataset.YOLOWeightedDataset")
    else:
        print("ℹ️ 未启用加权 DataLoader，使用默认 ultralytics.data.build.YOLODataset")

    cache_opt = args.cache if args.cache in ('ram', 'disk') else False
    # build.YOLODataset = YOLOWeightedDataset  # 旧的强制猴子补丁可以删掉或保留为注释

    print(f"🎯 使用 IoU 损失类型: {args.iou_type}")
    
    model.train(
        data=args.cfg,
        imgsz=args.img_size,
        epochs=args.epochs,
        batch=args.batch_size,
        device=args.device,
        workers=args.workers,
        cache=cache_opt,
        rect=args.rect,
        deterministic=args.deterministic,
        lr0=args.lr0,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        optimizer=args.optimizer,
        project=args.project,
        name=args.name,
        seed=args.seed,
        exist_ok=True,
        iou_type=args.iou_type,  # 添加 IoU 类型参数
        **augment_config,
        warmup_epochs=args.warmup_epochs,
        close_mosaic=args.close_mosaic,
        patience=50,
        save_period=10,
        amp=True,
        plots=True,
    )

    print(f"Training complete. Results saved to: {save_dir}")
