import torch
from torch import nn as nn
from torch.nn.functional import adaptive_avg_pool2d, interpolate


class NewConcat(nn.Module):
    """
    Improved fusion module to replace the standard Concat operation in YOLO series models.
    
    This module implements a lightweight feature fusion strategy with three key steps:
    1. Spatial alignment: Unifies feature map sizes using adaptive pooling and bilinear interpolation
    2. Channel alignment: Maps all inputs to a unified channel dimension using efficient convolutions
    3. Element-wise multiplication fusion: Suppresses noise and enhances common features across scales
    
    Advantages over standard Concat:
    - Reduces computational overhead by unifying output channels instead of stacking
    - Decreases information redundancy through multiplicative fusion
    - Improves feature quality by emphasizing multi-scale consistent features
    - More suitable for edge device deployment and real-time detection
    
    Note on gradient flow:
    - Element-wise multiplication is designed for typical YOLO architectures (2-3 branches)
    - BatchNorm layers help maintain gradient stability during training
    - For many branches (>4), consider enabling post_fusion for additional gradient paths
    
    Args:
        in_channels_list (list[int]): List of input channel counts for each branch
        out_channels (int): Unified output channel dimension after alignment and fusion
        kernel_size (int): Kernel size for channel alignment convolutions. Default: 3
        use_dw (bool): Whether to use depthwise separable convolution for efficiency. Default: True
        post_fusion (bool): Whether to apply lightweight post-fusion processing. Default: False
    """

    def __init__(self, in_channels_list, out_channels, kernel_size=3, use_dw=True, post_fusion=False):
        super().__init__()
        self.num_branches = len(in_channels_list)
        self.out_channels = out_channels
        self.use_dw = use_dw
        self.post_fusion = post_fusion

        # Create lightweight channel alignment modules for each input branch
        self.channel_aligners = nn.ModuleList()
        for in_channels in in_channels_list:
            if use_dw and in_channels == out_channels:
                # Use depthwise separable convolution when input/output channels match
                # This significantly reduces parameters and FLOPs
                aligner = nn.Sequential(
                    # Depthwise convolution: spatial feature extraction
                    nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                             padding=kernel_size // 2, groups=in_channels, bias=False),
                    nn.BatchNorm2d(in_channels),
                    nn.SiLU(inplace=True),
                    # Pointwise convolution: channel mixing
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.SiLU(inplace=True),
                )
            else:
                # Standard convolution for channel dimension transformation
                # Single conv layer is sufficient for channel alignment and local context extraction
                aligner = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                             padding=kernel_size // 2, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.SiLU(inplace=True),
                )
            self.channel_aligners.append(aligner)

        # Optional lightweight post-fusion processing
        # Only add if explicitly requested to keep the module minimal
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
        Forward pass: aligns features spatially and channel-wise, then fuses via element-wise multiplication.
        
        Args:
            x_list (list[torch.Tensor]): List of input feature maps. Each tensor has shape (B, C_i, H_i, W_i).
                                         Spatial dimensions (H_i, W_i) can differ across inputs.
        
        Returns:
            torch.Tensor: Fused feature map with shape (B, out_channels, H_target, W_target).
        """
        assert len(x_list) == self.num_branches, \
            f"Expected {self.num_branches} input feature maps, but got {len(x_list)}"

        # --- Step 1: Determine target spatial size ---
        # Strategy: Use the largest spatial size among all inputs as the alignment target
        # This preserves maximum spatial information
        target_size = None
        max_area = -1
        for x in x_list:
            h, w = x.shape[2], x.shape[3]
            area = h * w
            if area > max_area:
                max_area = area
                target_size = (h, w)

        # --- Step 2: Spatial alignment and channel alignment ---
        aligned_features = []
        for i, (x, aligner) in enumerate(zip(x_list, self.channel_aligners)):
            # Get current feature map dimensions
            _, _, h, w = x.shape

            # 2.1 Spatial alignment (if needed)
            if (h, w) != target_size:
                # Apply adaptive pooling (for downscaling) or bilinear interpolation (for upscaling)
                if h * w > target_size[0] * target_size[1]:
                    # Current feature map is larger -> use adaptive pooling to reduce size
                    x_resized = adaptive_avg_pool2d(x, target_size)
                else:
                    # Current feature map is smaller -> use bilinear interpolation to enlarge
                    x_resized = interpolate(x, size=target_size, mode='bilinear', align_corners=False)
            else:
                x_resized = x

            # 2.2 Channel alignment: transform to unified channel dimension
            x_aligned = aligner(x_resized)  # Shape: (B, out_channels, H_target, W_target)
            aligned_features.append(x_aligned)

        # --- Step 3: Element-wise multiplication fusion ---
        # This acts like an attention mechanism:
        # - Enhances features that are consistently activated across multiple scales
        # - Suppresses noise/background that appears only in single scales
        # - Achieves "feature purification" for improved fusion quality
        #
        # Note: Element-wise multiplication is the core design from the paper.
        # While it may cause gradient attenuation with many branches (>3-4),
        # this is acceptable for typical YOLO neck architectures which usually
        # fuse 2-3 feature maps. The BatchNorm layers before fusion help maintain
        # gradient flow by normalizing activations.
        fused = aligned_features[0]
        for feat in aligned_features[1:]:
            # Key operation: element-wise multiplication
            # All feature maps now have identical shape (B, out_channels, H_target, W_target)
            fused = fused * feat

        # --- Step 4: Optional post-fusion processing ---
        if self.post_fusion_conv is not None:
            fused = self.post_fusion_conv(fused)

        return fused
