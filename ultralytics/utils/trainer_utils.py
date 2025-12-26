# ultralytics/utils/trainer_utils.py (新建或添加到现有文件)

def print_training_config(args):
    """
    打印训练配置信息,突出显示 IoU 类型

    Args:
        args: 训练参数对象
    """
    from ultralytics.utils import LOGGER, colorstr

    # 分隔线
    LOGGER.info('\n' + '='*80)
    LOGGER.info(colorstr('bold', 'yellow', '🎯 YOLO11 Training Configuration'))
    LOGGER.info('='*80)

    # 基本配置 - 使用 getattr 安全获取属性
    LOGGER.info(colorstr('bold', '\n📁 Dataset & Model:'))
    LOGGER.info(f"  Model:      {getattr(args, 'model', 'yolo11n.pt')}")
    LOGGER.info(f"  Data:       {getattr(args, 'data', 'coco8.yaml')}")
    LOGGER.info(f"  Task:       {getattr(args, 'task', 'detect')}")

    # 训练参数 - 使用 getattr 安全获取属性
    LOGGER.info(colorstr('bold', '\n🔧 Training Parameters:'))
    LOGGER.info(f"  Epochs:     {getattr(args, 'epochs', 100)}")
    LOGGER.info(f"  Batch size: {getattr(args, 'batch', 16)}")
    LOGGER.info(f"  Image size: {getattr(args, 'imgsz', 640)}")
    LOGGER.info(f"  Device:     {getattr(args, 'device', 'cpu')}")

    # 🔥 IoU 配置 (突出显示)
    LOGGER.info(colorstr('bold', '\n📊 Loss Configuration:'))

    # 根据不同的 IoU 类型使用不同的颜色和说明
    iou_info = {
        'IoU': ('Standard IoU', 'white'),
        'GIoU': ('Generalized IoU', 'green'),
        'DIoU': ('Distance IoU', 'blue'),
        'CIoU': ('Complete IoU', 'cyan'),
        'ShapeIoU': ('Shape-aware IoU', 'magenta')
    }

    iou_type = getattr(args, 'iou_type', 'CIoU')
    desc, color = iou_info.get(iou_type, ('Unknown', 'white'))

    LOGGER.info(f"  IoU Type:   {colorstr(color, 'bold', f'{iou_type} ({desc})')}")
    LOGGER.info(f"  Box:        {getattr(args, 'box', 7.5)}")
    LOGGER.info(f"  Cls:        {getattr(args, 'cls', 0.5)}")
    LOGGER.info(f"  DFL:        {getattr(args, 'dfl', 1.5)}")

    # 优化器配置 - 使用 getattr 安全获取属性
    LOGGER.info(colorstr('bold', '\n⚡ Optimizer:'))
    LOGGER.info(f"  Optimizer:  {getattr(args, 'optimizer', 'auto')}")
    LOGGER.info(f"  LR0:        {getattr(args, 'lr0', 0.01)}")
    LOGGER.info(f"  Momentum:   {getattr(args, 'momentum', 0.937)}")
    LOGGER.info(f"  Weight decay: {getattr(args, 'weight_decay', 0.0005)}")

    LOGGER.info('\n' + '='*80 + '\n')


# 在 train.py 中使用:
# def main():
#     args = parse_args()
#
#     # 打印配置
#     print_training_config(args)
#
#     # 开始训练
#     trainer = YOLO(args.model)
#     trainer.train(**vars(args))

