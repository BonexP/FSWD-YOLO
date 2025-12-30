import torch
from torch import nn as nn
from torch.nn.functional import adaptive_avg_pool2d, interpolate


class NewConcat(nn.Module):
    """
    改进的融合模块，旨在取代YOLO系列模型中的标准Concat操作。Improved fusion module to replace the standard Concat operation in YOLO series models.

    本模块实现了一个轻量级的特征融合策略，包含三个关键步骤：This module implements a lightweight feature fusion strategy with three key steps:
    1. 空间对齐：使用自适应池化和双线性插值统一特征图尺寸 Spatial alignment: Unifies feature map sizes using adaptive pooling and bilinear interpolation
    2. 通道对齐：通过高效卷积将所有输入映射到统一的通道维 Channel alignment: Maps all inputs to a unified channel dimension using efficient convolutions
    3. 逐元素乘法融合：抑制噪声并强化跨尺度的一致特征 Element-wise multiplication fusion: Suppresses noise and enhances common features across scales

    相比标准Concat的优势：Advantages over standard Concat:
    - 通过统一输出通道而非堆叠来降低计算开销 Reduces computational overhead by unifying output channels instead of stacking
    - 通过乘法融合减少信息冗余 Decreases information redundancy through multiplicative fusion
    - 通过强调多尺度一致特征提升特征质量 Improves feature quality by emphasizing multi-scale consistent features
    - 更适合边缘设备部署与实时检测 More suitable for edge device deployment and real-time detection

    关于梯度流的说明：Note on gradient flow:
    - 逐元素乘法针对典型YOLO架构（2-3个分支）设计 Element-wise multiplication is designed for typical YOLO architectures (2-3 branches)
    - BatchNorm层有助于在训练中维持梯度稳定 BatchNorm layers help maintain gradient stability during training
    - 对于更多分支（>4）建议启用post_fusion以提供额外梯度路径 For many branches (>4), consider enabling post_fusion for additional gradient paths

    Args:
        in_channels_list (list[int]): 每个分支的输入通道数列表。List of input channel counts for each branch
        out_channels (int): 对齐与融合后统一的输出通道维度。Unified output channel dimension after alignment and fusion
        kernel_size (int): 通道对齐卷积的核大小。默认值为3。Kernel size for channel alignment convolutions. Default: 3
        use_dw (bool): 是否使用深度可分离卷积以提高效率。默认值为True。Whether to use depthwise separable convolution for efficiency. Default: True
        post_fusion (bool): 是否应用轻量级后融合处理。默认值为False。Whether to apply lightweight post-fusion processing. Default: False
    """

    def __init__(self, in_channels_list, out_channels, kernel_size=3, use_dw=True, post_fusion=False):
        super().__init__()
        self.num_branches = len(in_channels_list)
        self.out_channels = out_channels
        self.use_dw = use_dw
        self.post_fusion = post_fusion

        # 为每个输入分支创建轻量级通道对齐模块 Create lightweight channel alignment modules for each input branch
        self.channel_aligners = nn.ModuleList()
        for in_channels in in_channels_list:
            if use_dw and in_channels == out_channels:
                # 当输入/输出通道数匹配时使用深度可分离卷积 Use depthwise separable convolution when input/output channels match
                # 这样显著减少参数和FLOPs This significantly reduces parameters and FLOPs
                aligner = nn.Sequential(
                    # 深度卷积：提取空间特征 Depthwise convolution: spatial feature extraction
                    nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                             padding=kernel_size // 2, groups=in_channels, bias=False),
                    nn.BatchNorm2d(in_channels),
                    nn.SiLU(inplace=True),
                    # 逐点卷积：通道混合 Pointwise convolution: channel mixing
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.SiLU(inplace=True),
                )
            else:
                # 标准卷积用于通道维转换 Standard convolution for channel dimension transformation
                # 单层卷积足以完成通道对齐和局部上下文提取 Single conv layer is sufficient for channel alignment and local context extraction
                aligner = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                             padding=kernel_size // 2, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.SiLU(inplace=True),
                )
            self.channel_aligners.append(aligner)

        # 可选的轻量级后融合处理 Optional lightweight post-fusion processing
        # 仅在显式请求时添加以保持模块最小化 Only add if explicitly requested to keep the module minimal
        if post_fusion:
            self.post_fusion_conv = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(inplace=True),
            )
        else:
            self.post_fusion_conv = None

    def forward(self, x_list):
        """
        前向传播：先在空间和通道上对齐，再通过逐元素乘法融合。Forward pass: aligns features spatially and channel-wise, then fuses via element-wise multiplication.

        Args:
            x_list (list[torch.Tensor]): 输入特征图列表。每个张量形状为 (B, C_i, H_i, W_i)，不同分支的空间尺寸 (H_i, W_i) 可不同。List of input feature maps. Each tensor has shape (B, C_i, H_i, W_i). Spatial dimensions (H_i, W_i) can differ across inputs.

        Returns:
            torch.Tensor: 融合后的特征图，形状为 (B, out_channels, H_target, W_target)。Fused feature map with shape (B, out_channels, H_target, W_target).
        """
        assert len(x_list) == self.num_branches, \
            f"Expected {self.num_branches} input feature maps, but got {len(x_list)}"

        # --- 第一步：确定目标空间大小 ---
        # 策略：使用所有输入中的最大空间尺寸作为对齐目标 Strategy: Use the largest spatial size among all inputs as the alignment target
        # 这样能够保留最多的空间信息 This preserves maximum spatial information
        target_size = None
        max_area = -1
        for x in x_list:
            h, w = x.shape[2], x.shape[3]
            area = h * w
            if area > max_area:
                max_area = area
                target_size = (h, w)

        # --- 第二步：空间对齐与通道对齐 ---
        aligned_features = []
        for i, (x, aligner) in enumerate(zip(x_list, self.channel_aligners)):
            # 获取当前特征图尺寸 Get current feature map dimensions
            _, _, h, w = x.shape

            # 2.1 空间对齐（如有必要） Spatial alignment (if needed)
            if (h, w) != target_size:
                # 应用自适应池化（降采样）或双线性插值（上采样） Apply adaptive pooling (for downscaling) or bilinear interpolate
                if h * w > target_size[0] * target_size[1]:
                    # 当前特征图尺寸更大 -> 使用自适应池化缩小 Current feature map is larger -> use adaptive pooling to reduce size
                    x_resized = adaptive_avg_pool2d(x, target_size)
                else:
                    # 当前特征图尺寸更小 -> 使用双线性插值放大 Current feature map is smaller -> use bilinear interpolation to enlarge
                    x_resized = interpolate(x, size=target_size, mode='bilinear', align_corners=False)
            else:
                x_resized = x

            # 2.2 通道对齐：转换为统一的通道维度 Channel alignment: transform to unified channel dimension
            x_aligned = aligner(x_resized)  # Shape: (B, out_channels, H_target, W_target)
            aligned_features.append(x_aligned)

        # --- 第三步：逐元素乘法融合 ---
        # 该操作类似一种注意力机制：
        # - 强化在多个尺度上保持一致激活的特征
        # - 抑制仅在单一尺度出现的噪声或背景
        # - 实现“特征净化”，以提高融合质量
        # This acts like an attention mechanism:
        # - Enhances features that are consistently activated across multiple scales
        # - Suppresses noise/background that appears only in single scales
        # - Achieves "feature purification" for improved fusion quality

        # 注意：逐元素乘法是该设计的核心。
        # 虽然在分支较多（>3-4）时可能导致梯度衰减，但对于典型的YOLO neck架构（通常融合2-3个特征图）是可接受的。
        # 在融合前的BatchNorm层通过归一化激活有助于维持梯度流。
        # Note: Element-wise multiplication is the core design from the paper.
        # While it may cause gradient attenuation with many branches (>3-4),
        # this is acceptable for typical YOLO neck architectures which usually fuse 2-3 feature maps.
        # The BatchNorm layers before fusion help maintain gradient flow by normalizing activations.

        fused = aligned_features[0]
        for feat in aligned_features[1:]:
            # 关键操作：逐元素相乘（点对点乘）
            # Key operation: element-wise multiplication (pointwise product)
            # 所有特征图此时形状一致 (B, out_channels, H_target, W_target)
            # All feature maps now have identical shape (B, out_channels, H_target, W_target)
            fused = fused * feat

        # --- 第四步：可选的后融合处理 ---
        # --- Step 4: Optional post-fusion processing ---
        if self.post_fusion_conv is not None:
            fused = self.post_fusion_conv(fused)

        return fused
