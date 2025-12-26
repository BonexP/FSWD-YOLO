"""
自定义 IOU 损失函数集成示例
展示如何在 YOLO11 中添加新的 IoU 变体（如 EIoU, SIoU, WIoU 等）
"""

import math
import torch
import torch.nn as nn


# ============================================================================
# 示例 1: 在 metrics.py 中添加 EIoU (Efficient IoU) 计算
# ============================================================================

def bbox_eiou(box1: torch.Tensor, box2: torch.Tensor, xywh: bool = True, eps: float = 1e-7) -> torch.Tensor:
    """
    计算 EIoU (Efficient IoU)

    EIoU 改进了 CIoU，分别优化了宽度和高度的差异，而不是宽高比。

    Reference: https://arxiv.org/abs/2101.08158

    Args:
        box1: 预测框 tensor
        box2: 目标框 tensor
        xywh: 是否为 xywh 格式
        eps: 避免除零的小值

    Returns:
        EIoU 值
    """
    # 转换为 xyxy 格式
    if xywh:
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # 计算交集
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp_(0)

    # 并集
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union

    # EIoU 特有部分
    cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # 最小外接矩形宽度
    ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # 最小外接矩形高度
    c2 = cw.pow(2) + ch.pow(2) + eps  # 对角线平方

    # 中心距离
    rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2) +
            (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)) / 4

    # EIoU 的宽高损失（分别惩罚宽度和高度差异）
    rho_w2 = (b2_x2 - b2_x1 - (b1_x2 - b1_x1)).pow(2)
    rho_h2 = (b2_y2 - b2_y1 - (b1_y2 - b1_y1)).pow(2)
    cw2 = cw.pow(2) + eps
    ch2 = ch.pow(2) + eps

    return iou - (rho2 / c2 + rho_w2 / cw2 + rho_h2 / ch2)


def bbox_siou(box1: torch.Tensor, box2: torch.Tensor, xywh: bool = True, eps: float = 1e-7) -> torch.Tensor:
    """
    计算 SIoU (SCYLLA IoU)

    SIoU 考虑了向量角度、距离和形状损失。

    Reference: https://arxiv.org/abs/2205.12740

    Args:
        box1: 预测框 tensor
        box2: 目标框 tensor
        xywh: 是否为 xywh 格式
        eps: 避免除零的小值

    Returns:
        SIoU 值
    """
    # 转换为 xyxy 格式
    if xywh:
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # 计算交集
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp_(0)

    # 并集
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union

    # SIoU 特有部分
    # 1. 角度损失
    cx1, cy1 = (b1_x1 + b1_x2) / 2, (b1_y1 + b1_y2) / 2
    cx2, cy2 = (b2_x1 + b2_x2) / 2, (b2_y1 + b2_y2) / 2

    sigma = (cx2 - cx1).pow(2) + (cy2 - cy1).pow(2)
    sin_alpha = torch.abs(cy2 - cy1) / (sigma.sqrt() + eps)
    sin_beta = torch.abs(cx2 - cx1) / (sigma.sqrt() + eps)
    sin_alpha = torch.where(sin_alpha > math.sqrt(2) / 2, sin_beta, sin_alpha)
    angle_cost = 1 - 2 * (sin_alpha * math.pi / 4).sin().pow(2)

    # 2. 距离损失
    cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
    ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)
    rho_x = (cx2 - cx1).pow(2) / (cw.pow(2) + eps)
    rho_y = (cy2 - cy1).pow(2) / (ch.pow(2) + eps)
    gamma = 2 - angle_cost
    distance_cost = 1 - torch.exp(-gamma * rho_x) - torch.exp(-gamma * rho_y)

    # 3. 形状损失
    omiga_w = torch.abs(w1 - w2) / torch.max(w1, w2)
    omiga_h = torch.abs(h1 - h2) / torch.max(h1, h2)
    theta = 4
    shape_cost = (1 - torch.exp(-omiga_w)).pow(theta) + (1 - torch.exp(-omiga_h)).pow(theta)

    return iou - 0.5 * (distance_cost + shape_cost)


def bbox_wiou(box1: torch.Tensor, box2: torch.Tensor, xywh: bool = True, eps: float = 1e-7) -> torch.Tensor:
    """
    计算 WIoU (Wise IoU)

    WIoU 使用动态非单调聚焦机制。

    Reference: https://arxiv.org/abs/2301.10051

    Args:
        box1: 预测框 tensor
        box2: 目标框 tensor
        xywh: 是否为 xywh 格式
        eps: 避免除零的小值

    Returns:
        WIoU 值
    """
    # 转换为 xyxy 格式
    if xywh:
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # 计算交集
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp_(0)

    # 并集
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union

    # WIoU 特有部分 - 动态非单调聚焦机制
    cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
    ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)
    c2 = cw.pow(2) + ch.pow(2) + eps

    cx1, cy1 = (b1_x1 + b1_x2) / 2, (b1_y1 + b1_y2) / 2
    cx2, cy2 = (b2_x1 + b2_x2) / 2, (b2_y1 + b2_y2) / 2
    rho2 = ((cx2 - cx1).pow(2) + (cy2 - cy1).pow(2))

    # 智能梯度增益
    with torch.no_grad():
        beta = rho2 / c2
        alpha = 1 / (1 + torch.exp(-beta))

    return iou - alpha * (rho2 / c2)


# ============================================================================
# 示例 2: 自定义 BboxLoss 类
# ============================================================================

class CustomBboxLoss(nn.Module):
    """
    自定义边界框损失类，支持多种 IoU 变体
    """

    def __init__(self, reg_max: int = 16, iou_type: str = 'CIoU'):
        """
        Args:
            reg_max: DFL 的最大值
            iou_type: IoU 类型，可选: 'IoU', 'GIoU', 'DIoU', 'CIoU', 'EIoU', 'SIoU', 'WIoU'
        """
        super().__init__()
        from ultralytics.utils.loss import DFLoss

        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None
        self.iou_type = iou_type
        print(f"使用 {iou_type} 损失函数")

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
    ):
        """计算 IoU 和 DFL 损失"""
        from ultralytics.utils.metrics import bbox_iou
        from ultralytics.utils.tal import bbox2dist

        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)

        # 根据 iou_type 选择不同的 IoU 计算方法
        if self.iou_type == 'CIoU':
            iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        elif self.iou_type == 'DIoU':
            iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, DIoU=True)
        elif self.iou_type == 'GIoU':
            iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, GIoU=True)
        elif self.iou_type == 'EIoU':
            iou = bbox_eiou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False)
        elif self.iou_type == 'SIoU':
            iou = bbox_siou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False)
        elif self.iou_type == 'WIoU':
            iou = bbox_wiou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False)
        else:  # 标准 IoU
            iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False)

        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


# ============================================================================
# 示例 3: 如何在训练中使用自定义损失
# ============================================================================

def integrate_custom_loss_into_yolo():
    """
    集成自定义损失函数到 YOLO 的步骤说明
    """
    instructions = """
    # 步骤 1: 修改 ultralytics/utils/metrics.py
    # 将上面的 bbox_eiou, bbox_siou, bbox_wiou 函数添加到文件中
    
    # 步骤 2: 修改 ultralytics/utils/loss.py 中的 BboxLoss 类
    # 替换原有的 BboxLoss 为 CustomBboxLoss，或者添加 iou_type 参数
    
    # 修改前 (第108行附近):
    class BboxLoss(nn.Module):
        def __init__(self, reg_max: int = 16):
            super().__init__()
            self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None
    
    # 修改后:
    class BboxLoss(nn.Module):
        def __init__(self, reg_max: int = 16, iou_type: str = 'CIoU'):
            super().__init__()
            self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None
            self.iou_type = iou_type  # 添加这行
        
        def forward(self, ...):
            # 修改第128行附近的 IoU 计算
            if self.iou_type == 'EIoU':
                from .metrics import bbox_eiou
                iou = bbox_eiou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False)
            elif self.iou_type == 'SIoU':
                from .metrics import bbox_siou
                iou = bbox_siou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False)
            elif self.iou_type == 'WIoU':
                from .metrics import bbox_wiou
                iou = bbox_wiou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False)
            else:
                # 原有的 CIoU/DIoU/GIoU/IoU
                iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], 
                              xywh=False, CIoU=True)
            
            loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum
            # ...其余代码保持不变
    
    # 步骤 3: 修改 v8DetectionLoss.__init__() 传递 iou_type
    # 在第220行附近:
    class v8DetectionLoss:
        def __init__(self, model, tal_topk: int = 10):
            device = next(model.parameters()).device
            h = model.args
            
            # 从配置中读取 iou_type（需要在配置文件中添加）
            iou_type = getattr(h, 'iou_type', 'CIoU')  # 默认 CIoU
            
            # 修改这行
            self.bbox_loss = BboxLoss(m.reg_max, iou_type=iou_type).to(device)
    
    # 步骤 4: 在训练配置中添加 iou_type 参数
    # 在训练脚本中:
    from ultralytics import YOLO
    
    model = YOLO('yolo11n.yaml')
    
    # 方法 A: 通过训练参数传递
    results = model.train(
        data='coco.yaml',
        epochs=100,
        iou_type='EIoU'  # 指定使用 EIoU
    )
    
    # 方法 B: 修改配置文件 (例如 default.yaml)
    # 添加一行: iou_type: 'SIoU'
    """
    print(instructions)


# ============================================================================
# 示例 4: 测试不同 IoU 函数
# ============================================================================

def test_iou_functions():
    """测试不同的 IoU 计算函数"""
    # 创建测试数据
    box1 = torch.tensor([[100, 100, 200, 200]], dtype=torch.float32)  # xyxy 格式
    box2 = torch.tensor([[150, 150, 250, 250]], dtype=torch.float32)

    print("测试边界框:")
    print(f"Box1 (预测): {box1}")
    print(f"Box2 (目标): {box2}")
    print()

    # 导入原始的 bbox_iou
    try:
        from ultralytics.utils.metrics import bbox_iou as original_bbox_iou

        # 测试各种 IoU
        iou_standard = original_bbox_iou(box1, box2, xywh=False, CIoU=False, DIoU=False, GIoU=False)
        iou_giou = original_bbox_iou(box1, box2, xywh=False, GIoU=True)
        iou_diou = original_bbox_iou(box1, box2, xywh=False, DIoU=True)
        iou_ciou = original_bbox_iou(box1, box2, xywh=False, CIoU=True)

        print("IoU 计算结果:")
        print(f"标准 IoU:  {iou_standard.item():.4f}")
        print(f"GIoU:      {iou_giou.item():.4f}")
        print(f"DIoU:      {iou_diou.item():.4f}")
        print(f"CIoU:      {iou_ciou.item():.4f}")

    except ImportError:
        print("无法导入 ultralytics.utils.metrics，使用自定义函数测试")

    # 测试自定义 IoU
    iou_eiou = bbox_eiou(box1, box2, xywh=False)
    iou_siou = bbox_siou(box1, box2, xywh=False)
    iou_wiou = bbox_wiou(box1, box2, xywh=False)

    print(f"EIoU:      {iou_eiou.item():.4f}")
    print(f"SIoU:      {iou_siou.item():.4f}")
    print(f"WIoU:      {iou_wiou.item():.4f}")
    print()

    # 计算损失 (1 - IoU)
    print("对应的损失值 (1 - IoU):")
    print(f"CIoU Loss: {(1 - iou_ciou).item():.4f}")
    print(f"EIoU Loss: {(1 - iou_eiou).item():.4f}")
    print(f"SIoU Loss: {(1 - iou_siou).item():.4f}")
    print(f"WIoU Loss: {(1 - iou_wiou).item():.4f}")


# ============================================================================
# 主函数
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("YOLO11 自定义 IoU 损失函数集成示例")
    print("=" * 80)
    print()

    print("📝 集成步骤说明:")
    print("-" * 80)
    integrate_custom_loss_into_yolo()
    print()

    print("🧪 测试不同的 IoU 函数:")
    print("-" * 80)
    test_iou_functions()
    print()

    print("✅ 完成！")
    print()
    print("📌 推荐的修改方式:")
    print("1. 将新的 IoU 函数添加到 ultralytics/utils/metrics.py")
    print("2. 修改 ultralytics/utils/loss.py 中的 BboxLoss 类添加 iou_type 参数")
    print("3. 在训练时通过参数指定使用哪种 IoU: model.train(iou_type='EIoU')")
    print()
    print("📖 参考文献:")
    print("- EIoU: https://arxiv.org/abs/2101.08158")
    print("- SIoU: https://arxiv.org/abs/2205.12740")
    print("- WIoU: https://arxiv.org/abs/2301.10051")

