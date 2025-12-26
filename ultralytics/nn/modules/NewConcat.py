from torch import nn as nn
from torch.nn.functional import adaptive_avg_pool2d, interpolate


class NewConcat(nn.Module):
    """
    改进的融合模块，用于替换YOLO系列中的标准Concat操作。
    原理：空间对齐 -> 通道对齐 -> 逐元素乘法融合。
    优点：1) 通过统一输出通道数降低计算复杂度；2) 利用乘法抑制噪声、增强共有特征。

    Args:
        in_channels_list (list[int]): 输入特征图的通道数列表，长度N对应输入分支数。
        out_channels (int): 对齐并融合后的统一输出通道数。
        kernel_size (int): 通道对齐网络中使用卷积核的大小。默认为3。
        activation (nn.Module): 激活函数，默认为 nn.SiLU()。
    """

    def __init__(self, in_channels_list, out_channels, kernel_size=3, activation=nn.SiLU()):
        super().__init__()
        self.num_branches = len(in_channels_list)
        self.out_channels = out_channels

        # 为每个输入分支创建一个独立的通道对齐模块
        self.channel_aligners = nn.ModuleList()
        for in_channels in in_channels_list:
            # 构建一个小型卷积网络进行通道变换和特征提取
            # 论文中提到“multiple convolutional layers”，这里用两层卷积为例
            aligner = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
                nn.BatchNorm2d(out_channels),
                activation,
                nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
                nn.BatchNorm2d(out_channels),
                activation,
            )
            self.channel_aligners.append(aligner)

        # 可选的：在融合后可以再加一个轻量的卷积进行后处理（非论文必须，但实践中常见）
        self.post_fusion_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            activation,
        )

    def forward(self, x_list):
        """
        Args:
            x_list (list[Tensor]): 输入特征图列表。每个Tensor形状为 (B, C_i, H_i, W_i)。
                                   允许空间尺寸不同。
        Returns:
            fused_feat (Tensor): 融合后的特征图，形状为 (B, out_channels, H_target, W_target)。
        """
        assert len(x_list) == self.num_branches, \
            f"输入特征图数量({len(x_list)})与初始化数量({self.num_branches})不符"

        # --- 1. 确定目标空间尺寸 ---
        # 策略：选择所有输入中空间尺寸最大的作为对齐目标
        target_size = None
        max_area = -1
        for x in x_list:
            h, w = x.shape[2], x.shape[3]
            area = h * w
            if area > max_area:
                max_area = area
                target_size = (h, w)

        # --- 2. 空间对齐与通道对齐 ---
        aligned_features = []
        for i, (x, aligner) in enumerate(zip(x_list, self.channel_aligners)):
            # 获取当前特征图尺寸
            _, _, h, w = x.shape

            # 2.1 空间对齐（如果需要）
            if (h, w) != target_size:
                # 论文：Adaptive Pooling (缩小) 和 Bilinear Interpolation (放大)
                # 使用 adaptive_avg_pool2d 和 interpolate 来实现
                if h * w > target_size[0] * target_size[1]:
                    # 当前特征图更大 -> 使用自适应池化缩小
                    # 注意：自适应池化可能导致信息丢失，但符合论文描述
                    x_resized = adaptive_avg_pool2d(x, target_size)
                else:
                    # 当前特征图更小 -> 使用双线性插值放大
                    x_resized = interpolate(x, size=target_size, mode='bilinear', align_corners=False)
            else:
                x_resized = x

            # 2.2 通道对齐
            x_aligned = aligner(x_resized)  # 形状变为 (B, out_channels, H_target, W_target)
            aligned_features.append(x_aligned)

        # --- 3. 逐元素乘法融合 ---
        # 初始化融合结果为第一个对齐后的特征图
        fused = aligned_features[0]
        for feat in aligned_features[1:]:
            # 关键步骤：逐元素乘法
            fused = fused * feat  # 所有特征图此时形状完全相同

        # --- 4. 后处理（可选）---
        fused = self.post_fusion_conv(fused)

        return fused
