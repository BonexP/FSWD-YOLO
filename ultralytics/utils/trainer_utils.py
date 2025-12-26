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

    # 基本配置
    LOGGER.info(colorstr('bold', '\n📁 Dataset & Model:'))
    LOGGER.info(f"  Model:      {args.model}")
    LOGGER.info(f"  Data:       {args.data}")
    LOGGER.info(f"  Task:       {args.task}")

    # 训练参数
    LOGGER.info(colorstr('bold', '\n🔧 Training Parameters:'))
    LOGGER.info(f"  Epochs:     {args.epochs}")
    LOGGER.info(f"  Batch size: {args.batch}")
    LOGGER.info(f"  Image size: {args.imgsz}")
    LOGGER.info(f"  Device:     {args.device}")

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
    LOGGER.info(f"  Box:        {args.box}")
    LOGGER.info(f"  Cls:        {args.cls}")
    LOGGER.info(f"  DFL:        {args.dfl}")

    # 优化器配置
    LOGGER.info(colorstr('bold', '\n⚡ Optimizer:'))
    LOGGER.info(f"  Optimizer:  {args.optimizer}")
    LOGGER.info(f"  LR0:        {args.lr0}")
    LOGGER.info(f"  Momentum:   {args.momentum}")
    LOGGER.info(f"  Weight decay: {args.weight_decay}")

    LOGGER.info('\n' + '='*80 + '\n')


# 在 train.py 中使用:
def main():
    args = parse_args()

    # 打印配置
    print_training_config(args)

    # 开始训练
    trainer = YOLO(args.model)
    trainer.train(**vars(args))