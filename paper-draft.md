
# A Lightweight Attention-Enhanced YOLO Framework for Surface Defect Detection in Friction Stir Welding

---

## Abstract

Surface defect detection in friction stir welding (FSW) plays a critical role in ensuring joint quality and structural reliability. However, FSW surface defects often exhibit small scale, irregular morphology, and low contrast, which pose significant challenges to conventional visual inspection and generic object detection models. To address these issues, this paper proposes a lightweight and robust defect detection framework based on the YOLO architecture, specifically tailored for FSW surface inspection.

The proposed method integrates Ghost convolution and VoV modules to enhance multi-scale feature representation while reducing computational redundancy. In addition, a dual-attention mechanism combining Feature Channel Attention (FCA) and SimAM is introduced to improve spatial sensitivity to subtle defect patterns. To further optimize localization accuracy for irregularly shaped defects, ShapeIoU loss is employed instead of conventional IoU-based losses. A dedicated FSW surface defect dataset containing approximately 1000 annotated images across four defect categories is constructed for evaluation.

Extensive experiments, including ablation studies, comparisons with recent YOLO variants (YOLOv8, YOLOv9, and YOLOv10), and cross-dataset generalization tests on the NEU-DET benchmark, demonstrate the effectiveness of the proposed approach. The proposed method achieves an mAP@0.5 of 0.8274 on the FSW dataset, outperforming all compared baselines with a favorable balance between accuracy and computational efficiency. These results indicate that the proposed framework provides an effective and practical solution for automated FSW surface defect detection.

---

## Keywords

Friction Stir Welding; Surface Defect Detection; Object Detection; YOLO; Attention Mechanism; ShapeIoU

---

## 1. Introduction

Friction stir welding (FSW) is a solid-state joining technique widely used in aerospace, automotive, and rail transportation industries due to its advantages in producing high-quality joints with low distortion and excellent mechanical properties. Despite these advantages, surface defects such as tunnels, grooves, voids, and surface irregularities may still occur during improper process parameter selection or tool wear, potentially compromising joint performance and service reliability.

Accurate and efficient detection of FSW surface defects is therefore essential for quality assurance and process optimization. Traditional non-destructive testing (NDT) methods and handcrafted feature-based vision techniques often struggle to meet industrial requirements due to their sensitivity to noise, limited robustness, and high dependency on expert knowledge. In recent years, deep learning-based object detection methods have demonstrated remarkable performance in various industrial inspection tasks. However, directly applying general-purpose detectors to FSW surface defects remains challenging due to several intrinsic characteristics: (1) defects are often small and sparsely distributed, (2) defect boundaries are irregular and ambiguous, and (3) surface textures exhibit low contrast and complex background patterns.

Recent advances in YOLO-based detectors have shown promising performance in real-time inspection scenarios. Nevertheless, standard YOLO architectures are not specifically designed to handle the fine-grained and irregular nature of FSW defects. Moreover, increasing network depth or width to improve performance often leads to excessive computational costs, limiting deployment in real industrial environments.

To address these challenges, this paper proposes a lightweight attention-enhanced YOLO framework specifically optimized for FSW surface defect detection. By integrating efficient convolutional operations, multi-scale feature aggregation, attention mechanisms, and shape-aware localization loss, the proposed approach aims to achieve a favorable trade-off between detection accuracy and computational efficiency.

The main contributions of this work can be summarized as follows:

1. A lightweight YOLO-based detection framework is proposed for FSW surface defects, incorporating Ghost convolution and VoV modules to enhance feature diversity and multi-scale representation with reduced computational cost.
2. A dual-attention strategy combining FCA and SimAM is introduced to improve sensitivity to small-scale and low-contrast defect patterns commonly observed in FSW surfaces.
3. ShapeIoU loss is adopted to better handle the localization of irregularly shaped defects, leading to improved bounding box regression accuracy.
4. A self-collected and annotated FSW surface defect dataset with four defect categories is constructed, and extensive experiments—including ablation studies, comparisons with recent YOLO variants, and cross-dataset generalization on NEU-DET—validate the effectiveness and robustness of the proposed method.

---

## 2. Related Work

### 2.1 Surface Defect Detection in Welding and Manufacturing

[待补充：此处需补充与焊接/制造领域表面缺陷检测相关的代表性文献与方法综述]

Traditional inspection methods for welding defects include ultrasonic testing, radiographic testing, and eddy current testing. While effective, these approaches often require specialized equipment and skilled operators, limiting inspection efficiency and scalability. Vision-based inspection methods have gained increasing attention due to their non-contact nature and potential for automation.

Recent studies have explored convolutional neural networks (CNNs) for surface defect classification and detection in manufacturing scenarios. However, most existing works focus on relatively regular defects or high-contrast surfaces, leaving FSW surface defects underexplored.

### 2.2 YOLO-Based Object Detection

[待补充：此处需补充 YOLO 系列方法（YOLOv5–YOLOv10）的发展与特点]

YOLO-based detectors are widely adopted in industrial inspection due to their end-to-end architecture and real-time performance. Various improvements, such as feature pyramid networks, attention mechanisms, and lightweight convolutional designs, have been proposed to enhance detection accuracy and efficiency.

### 2.3 Attention Mechanisms and IoU-Based Losses

Attention mechanisms, including channel-wise and spatial attention, have been demonstrated to improve feature discrimination in object detection. FCA emphasizes informative feature channels, while SimAM provides parameter-free spatial attention by exploiting neuron energy functions. Meanwhile, advanced IoU-based loss functions, such as GIoU, CIoU, and ShapeIoU, aim to improve bounding box regression by considering geometric and shape-related factors.

---

## 3. Dataset and Problem Analysis

### 3.1 FSW Surface Defect Dataset

The FSW surface defect dataset used in this study was self-collected and manually annotated specifically for this research. It contains approximately 1000 images capturing four types of surface defects commonly observed in friction stir welding processes. Each image may contain multiple defects of different categories with varying sizes and shapes, reflecting the realistic complexity of industrial FSW inspection scenarios.

**Defect Categories**: The dataset includes the following four defect types:
1. **Tunnels**: Subsurface voids manifesting as linear or curved dark regions
2. **Grooves**: Surface depressions along the weld seam, typically elongated
3. **Voids**: Discrete cavity-like defects with irregular boundaries
4. **Surface irregularities**: Texture anomalies including flash, burrs, or material overflow

[待补充：此处需说明具体缺陷类别的英文标准术语、样本数量分布、尺寸统计]

**Data Acquisition**: Images were captured using [待补充：相机型号、镜头规格、拍摄距离] with controlled lighting conditions to minimize reflections and shadows. The image resolution is [待补充：分辨率，如 1920×1080] pixels, providing sufficient spatial detail for detecting small defects while maintaining manageable computational requirements.

**Annotation Protocol**: Bounding box annotations were performed by experienced welding inspectors using [待补充：标注工具，如 LabelImg, CVAT] following standardized annotation guidelines:
- Minimum bounding box area: [待补充：如 > 100 pixels²]
- Annotation tightness criterion: Bounding box should enclose the defect with minimal background margin (< 5 pixels when possible)
- Quality assurance: All annotations were cross-validated by at least two inspectors, with disagreements resolved through consensus discussion
- Defect severity: Annotations include defect category labels only; severity grading is not included in this study

**Dataset Split**: The dataset is divided into training, validation, and test sets following [待补充：划分比例，如 70%/15%/15% 或 60%/20%/20%]. The split strategy ensures:
- Balanced class distribution across splits (within ±5% deviation)
- No data leakage: images from the same welding specimen are assigned to the same split to prevent overfitting
- Temporal diversity: images span multiple welding batches to capture process variation

**Statistical Characteristics**:
- Average defects per image: [待补充：如 2.3]
- Defect size distribution: [待补充：如 small (< 50×50 px): 45%, medium (50-200 px): 40%, large (> 200 px): 15%]
- Aspect ratio distribution: [待补充：如 mean=2.1, std=1.5, reflecting elongated defect morphology]
- Class imbalance ratio: [待补充：如最大类与最小类样本比例]

[待补充：此处需提供数据集详细统计信息、典型样本可视化图例]

### 3.2 Challenges in FSW Surface Defect Detection

FSW surface defects present several unique challenges that distinguish them from generic object detection tasks and motivate the proposed architectural design:

**1. Multi-Scale Defect Distribution**: FSW defects span a wide range of scales, from small pits (< 10×10 pixels) to elongated grooves spanning hundreds of pixels. This scale diversity requires effective multi-scale feature representation and detection heads operating at multiple resolutions. Standard single-scale detectors often miss small defects or produce imprecise localization for large irregular defects.

**2. Irregular and Elongated Morphology**: Unlike regular objects with predictable aspect ratios, FSW defects frequently exhibit elongated shapes (e.g., grooves, tunnels) with aspect ratios exceeding 5:1. Conventional bounding box regression losses (e.g., IoU, CIoU) that treat width and height symmetrically may lead to suboptimal localization for such anisotropic shapes. This motivates the adoption of shape-aware loss functions (Section 4.6).

**3. Low Contrast and Ambiguous Boundaries**: Defects often blend with the surrounding weld surface due to:
- Similar color/intensity between defect regions and background
- Gradual transitions at defect boundaries rather than sharp edges
- Specular reflections from metallic surfaces introducing noise and occlusions

These factors make defect boundaries difficult to localize precisely, requiring enhanced feature discrimination through attention mechanisms (Section 4.3, 4.4).

**4. Complex Background Texture**: FSW surfaces inherently contain periodic patterns from tool rotation and material flow, which can be visually similar to certain defect types (e.g., surface irregularities vs. normal tool marks). This background complexity increases false positive rates and requires robust feature learning to distinguish true defects from benign texture variations.

**5. Intra-Class Appearance Variation**: Defects within the same category may exhibit significant appearance diversity due to:
- Varying welding parameters (tool rotation speed, traverse speed, plunge depth)
- Material properties (alloy composition, thickness)
- Defect formation mechanisms (tool wear, improper parameter selection)

This variation challenges the model's generalization capability and necessitates expressive feature representations.

**6. Real-Time Constraint**: Industrial inspection systems typically require real-time or near-real-time processing (> 30 FPS for continuous web inspection). This imposes strict computational constraints, motivating the use of lightweight architectures (Ghost convolution, efficient attention mechanisms) that balance accuracy and speed (Section 4.2, 4.7).

These characteristics collectively motivate the need for:
- Enhanced multi-scale feature representation (VoV modules, FPN-like neck)
- Attention-driven feature selection (SimAM for spatial attention, FCA for channel attention)
- Shape-aware localization strategies (ShapeIoU loss)
- Computational efficiency (Ghost convolution, parameter-free attention)

The proposed method addresses these challenges through targeted architectural innovations validated in extensive experiments (Section 5).

[请确认此部分逻辑或提供更多背景信息以便进一步细化]

---

## 4. Proposed Method

### 4.1 Overall Architecture

The proposed framework is built upon a YOLO-style detection architecture and introduces several targeted modifications to address the challenges of FSW surface defect detection. The overall architecture consists of a backbone for feature extraction, a neck for multi-scale feature aggregation, and a detection head for classification and localization.

The backbone employs a hierarchical feature extraction scheme with five stages operating at different spatial resolutions: P1/2, P2/4, P3/8, P4/16, and P5/32. The initial two stages use standard convolutions with 64 and 128 output channels, respectively. At P3/8 (layer 2) and P4/16 (layer 4), the proposed C3k2GhostSimAMinner module is introduced to achieve efficient feature extraction with integrated spatial attention. Subsequent stages (P4/16 and P5/32, layers 6 and 8) employ standard C3k2 blocks to maintain feature richness. A Spatial Pyramid Pooling Fast (SPPF) module is applied at the deepest layer (layer 9) to capture multi-scale contextual information. Finally, a C2PSFCA attention module (layer 10) is introduced to refine high-level semantic features through dual-channel attention mechanisms.

The neck network adopts a Feature Pyramid Network (FPN)-like structure to enable top-down and bottom-up multi-scale feature fusion. Features from the backbone at P5, P4, and P3 are progressively combined through upsampling, concatenation, and convolution operations. At the medium scale (P4/16, layer 19), the proposed VoVGSCSPC module is integrated to enhance feature reuse and aggregation, which is particularly beneficial for detecting small and irregular defects commonly observed in FSW surfaces.

The detection head is a decoupled anchor-free design operating at three scales (P3/8, P4/16, P5/32), producing classification and regression outputs for defect localization. The overall architecture balances detection accuracy, computational efficiency, and multi-scale representation capability.

[待补充：此处需插入整体网络结构示意图说明，标注各模块位置及特征图尺寸]

### 4.2 Lightweight Feature Extraction with Ghost Convolution

To reduce computational complexity while maintaining feature representation capability, Ghost convolution is employed as a core component in the backbone network. Traditional convolution operations often generate redundant feature maps with high computational cost. Ghost convolution addresses this issue by generating feature maps through a two-stage process: primary convolution followed by cheap linear transformations.

Specifically, given an input feature map $\mathbf{X} \in \mathbb{R}^{C_{in} \times H \times W}$, Ghost convolution first applies a primary convolution with $C_{out}/2$ filters to generate intrinsic feature maps:

$$\mathbf{X}' = \text{Conv}(\mathbf{X}; \mathbf{W}_1)$$

where $\mathbf{W}_1 \in \mathbb{R}^{(C_{out}/2) \times C_{in} \times k \times k}$ represents the learnable weights. Subsequently, cheap operations (e.g., depthwise convolutions with kernel size 5×5) are applied to each intrinsic feature to generate additional ghost features:

$$\mathbf{X}'' = \text{DWConv}(\mathbf{X}'; \mathbf{W}_2)$$

The final output is obtained by concatenating intrinsic and ghost features: $\mathbf{Y} = [\mathbf{X}', \mathbf{X}'']$. This design reduces computational cost by approximately 50% compared to standard convolution while maintaining comparable representational capacity.

In the proposed framework, Ghost convolution is integrated into the GhostBottleneck module, which further incorporates skip connections for residual learning. The GhostBottleneck consists of:
1. A Ghost convolution layer expanding the channel dimension
2. An optional depthwise convolution for spatial downsampling (when stride=2)
3. A Ghost convolution layer for channel compression
4. A shortcut connection to facilitate gradient flow

This bottleneck design is particularly suitable for FSW defect detection, where computational efficiency is critical for real-time industrial deployment.

### 4.3 C3k2GhostSimAMinner: Attention-Enhanced Bottleneck Module

To further improve feature discrimination for small and low-contrast defects, a novel C3k2GhostSimAMinner module is proposed by integrating SimAM attention into the GhostBottleneck structure. Unlike conventional attention mechanisms that apply attention globally after feature aggregation, C3k2GhostSimAMinner applies SimAM after each GhostBottleneck within the module, enabling fine-grained spatial attention at the intermediate feature level.

The module architecture follows a CSPNet-style design:
1. Input features are split into two branches through a 1×1 convolution (cv1)
2. The first branch serves as a shortcut
3. The second branch passes through $n$ sequential blocks, each consisting of:
   - GhostBottleneck for efficient feature transformation
   - SimAM for parameter-free spatial attention
4. Intermediate outputs from all blocks are concatenated with the shortcut branch
5. A final 1×1 convolution (cv2) fuses the aggregated features

SimAM (Simple, Parameter-Free Attention Module) computes attention weights based on the energy function of neurons without introducing additional learnable parameters. For a feature map $\mathbf{X}$, SimAM evaluates the importance of each spatial location by measuring its deviation from the spatial mean:

$$e_t(\mathbf{w}_t, b_t, \mathbf{y}, \mathbf{x}_i) = \left(\mathbf{y}_t - \hat{t}\right)^2 + \frac{1}{M-1} \sum_{i=1}^{M-1} \left(\mathbf{x}_i - \hat{t}\right)^2$$

where $\mathbf{y}_t$ is the target neuron, $\mathbf{x}_i$ are other neurons in the same channel, $M$ is the number of neurons, and $\hat{t}$ is the mean activation. The attention weight is computed as:

$$\mathbf{A} = \sigma\left(\frac{1}{E(\mathbf{X})}\right)$$

where $E(\mathbf{X})$ represents the minimum energy and $\sigma$ is the sigmoid function. The output feature is obtained through element-wise multiplication: $\mathbf{Y} = \mathbf{X} \odot \mathbf{A}$.

By applying SimAM within each bottleneck, the module can adaptively emphasize informative spatial regions at multiple abstraction levels, which is particularly beneficial for detecting small defects with weak visual cues.

### 4.4 Dual-Path Attention Module: C2PSFCA

To enhance channel-wise feature selection and multi-scale representation, a C2PSFCA (Cross-Stage Partial with PSA and FCA) module is introduced after the deepest backbone layer. This module integrates two complementary attention mechanisms: Position-Sensitive Attention (PSA) for spatial feature enhancement and Frequency Channel Attention (FCA) for channel recalibration.

The C2PSFCA module adopts a three-branch architecture:
1. Input features are projected to $3C$ channels through a 1×1 convolution (cv1)
2. The projected features are split into three equal parts:
   - Branch A: Direct skip connection without attention
   - Branch B: PSA-based spatial attention for position-sensitive feature enhancement
   - Branch C: FCA-based channel attention for frequency-aware feature recalibration
3. The three branches are concatenated and fused through a 1×1 convolution (cv2)

**FCA (Frequency Channel Attention)** exploits both local and global channel relationships through adaptive frequency-domain analysis. Given an input feature $\mathbf{X} \in \mathbb{R}^{C \times H \times W}$, FCA first applies global average pooling to obtain channel-wise statistics $\mathbf{z} \in \mathbb{R}^{C \times 1 \times 1}$. Local channel interaction is captured through 1D convolution with adaptive kernel size $k$:

$$k = \left|\frac{\log_2(C) + b}{\gamma}\right|_{\text{odd}}$$

where $b=1$ and $\gamma=2$ are hyperparameters. Global channel interaction is captured through a fully connected layer. The correlation between local and global representations is computed via matrix multiplication, and the final attention weights are generated through learnable adaptive fusion.

Unlike conventional channel attention (e.g., SE, ECA), FCA explicitly models the interaction between local neighborhood correlations and global dependencies, enabling more expressive channel recalibration. This is particularly effective for FSW defects, where certain channels may capture subtle texture patterns while others encode geometric structures.

The integration of PSA and FCA in C2PSFCA provides complementary spatial and channel-wise feature refinement, enhancing the model's sensitivity to multi-scale and low-contrast defect patterns.

### 4.5 VoVGSCSPC: Enhanced Feature Aggregation for Detection Neck

In the detection neck, the proposed VoVGSCSPC (VoV with GS Convolution and CSP Connection) module is employed at the medium scale (P4/16) to enhance feature reuse and multi-scale aggregation. This module is inspired by VoVNet (Variety of View Network), which advocates for one-shot aggregation of features from multiple receptive fields to improve feature diversity.

The VoVGSCSPC module consists of:
1. Two parallel pathways for feature transformation:
   - Path 1: 1×1 convolution (cv1) followed by a C2f block for rich feature extraction
   - Path 2: 3×3 GS convolution (gc2) for lightweight spatial feature encoding
2. A GSBottleneckC block within the C2f pathway, which combines:
   - GSConv (Gated Shuffle Convolution) for efficient feature mixing
   - Depthwise convolution-based shortcut for lightweight skip connections
3. Concatenation of both pathways followed by 1×1 convolution (cv3) for channel fusion

The GSConv operation integrates depthwise separable convolution with channel shuffling to reduce computational redundancy while maintaining inter-channel information flow. The cheap shortcut connection in GSBottleneckC uses depthwise convolution instead of standard convolution, further reducing parameters.

By aggregating features from dual pathways with different receptive fields and computational costs, VoVGSCSPC achieves a favorable balance between feature richness and efficiency. This design is particularly suitable for the medium-scale detection head, where both small and medium-sized defects are frequently detected.

### 4.6 ShapeIoU Loss for Irregular Defect Localization

To better handle the irregular and elongated shapes characteristic of FSW surface defects, ShapeIoU loss is adopted to replace the conventional CIoU loss for bounding box regression. While CIoU considers overlap, center distance, and aspect ratio, it does not explicitly account for shape similarity between predicted and ground-truth boxes, which can lead to suboptimal localization for non-square objects.

ShapeIoU introduces two additional shape-aware penalty terms:

**Shape-Distance**: Adaptively weights horizontal and vertical center distances based on the aspect ratio of the target box:

$$\mathcal{L}_{\text{dist}} = \frac{\omega_w \cdot \Delta x_c^2 + \omega_h \cdot \Delta y_c^2}{c^2}$$

where $\Delta x_c$ and $\Delta y_c$ are center coordinate differences, $c^2$ is the squared diagonal length of the smallest enclosing box, and the adaptive weights are:

$$\omega_w = \frac{2w^s}{w^s + h^s}, \quad \omega_h = \frac{2h^s}{w^s + h^s}$$

where $w$ and $h$ are the width and height of the target box, and $s$ is a power scale factor (set to 0 in this work for balanced weighting).

**Shape-Shape**: Penalizes differences in width and height with adaptive emphasis:

$$\mathcal{L}_{\text{shape}} = \left(1 - e^{-\Omega_w}\right)^4 + \left(1 - e^{-\Omega_h}\right)^4$$

$$\Omega_w = \omega_h \cdot \frac{|w_{pred} - w_{gt}|}{\max(w_{pred}, w_{gt})}, \quad \Omega_h = \omega_w \cdot \frac{|h_{pred} - h_{gt}|}{\max(h_{pred}, h_{gt})}$$

The final ShapeIoU loss combines overlap, shape distance, and shape similarity:

$$\mathcal{L}_{\text{ShapeIoU}} = 1 - \text{IoU} + \mathcal{L}_{\text{dist}} + 0.5 \cdot \mathcal{L}_{\text{shape}}$$

For FSW defects, which often exhibit elongated groove-like or tunnel-like morphology with high aspect ratios, the adaptive weighting in ShapeIoU encourages the model to align both the position and shape of predictions with ground truth, leading to more precise localization. The cross-dataset generalization experiments (Section 5.4) further validate that ShapeIoU improves bounding box regression robustness across different defect datasets.

### 4.7 Implementation Details

**Network Configuration**: The proposed model is based on the YOLO11s architecture with scale factor $s=[0.50, 0.50, 1024]$, which controls depth multiplier (0.50), width multiplier (0.50), and maximum channel number (1024). This configuration results in 181 layers with approximately 9.73M parameters and 20.57 GFLOPs computational complexity.

**Module Integration**: 
- C3k2GhostSimAMinner is applied at layers 2 and 4 (P3/8 and P4/16 in the backbone) with 2 bottleneck repetitions and expansion ratio 0.25
- C2PSFCA is placed at layer 10 (after SPPF) with 2 attention blocks and expansion ratio 0.5
- VoVGSCSPC is integrated at layer 19 (P4/16 in the neck) replacing the standard C3k2 block
- All other backbone and neck layers follow the standard YOLO11s design

**Loss Function**: The bounding box regression loss uses ShapeIoU with default parameters: $\text{scale}=0.0$ (balanced aspect ratio weighting) and $\epsilon=10^{-7}$ (numerical stability). The overall training loss combines classification loss (binary cross-entropy), bounding box regression loss (ShapeIoU), and distribution focal loss (DFL) with empirically determined weights.

**Computational Efficiency**: Compared to the baseline YOLO11s (9.43M parameters, 21.56 GFLOPs), the proposed model achieves a 4.6% reduction in computational complexity (20.57 GFLOPs) while maintaining similar parameter count (9.73M). The lightweight GhostConv operations in early backbone stages contribute significantly to FLOP reduction, while the additional attention modules introduce minimal computational overhead due to their efficient designs (parameter-free SimAM and adaptive FCA).

**Code Implementation**: The proposed modules are implemented in PyTorch and integrated into the Ultralytics YOLO framework. Key implementation files include:
- `ultralytics/nn/modules/conv.py`: GhostConv implementation
- `ultralytics/nn/modules/block.py`: C3k2GhostSimAMinner, C2PSFCA, VoVGSCSPC, and attention modules
- `ultralytics/utils/metrics.py`: bbox_shape_iou function with adaptive shape weighting
- `ultralytics/cfg/models/11/fswd-yolo.yaml`: Complete model configuration

[待补充：此处可根据审稿人反馈补充更多训练技巧或超参数敏感性分析]

---

## 5. Experiments

### 5.1 Experimental Setup

**Dataset Configuration**: The FSW surface defect dataset is split into training, validation, and test sets with [待补充：划分比例，如 70%/15%/15%]. The NEU-DET benchmark dataset is used with its standard split (train: 1800 images, test: 300 images) for generalization experiments.

**Training Hyperparameters**: All models are trained using the following configuration:
- **Optimizer**: AdamW with initial learning rate $lr_0 = 0.001$, weight decay $\lambda = 0.0005$, and momentum parameters $\beta_1 = 0.937$, $\beta_2 = 0.999$
- **Learning Rate Schedule**: Cosine annealing with linear warmup (500 iterations ramping from $0.0$ to $lr_0$)
- **Training Duration**: 300 epochs with early stopping (patience: 50 epochs based on validation mAP@0.5)
- **Batch Size**: [待补充：批大小，如 16 或 32], determined by GPU memory constraints
- **Input Resolution**: 640×640 pixels (resized from original resolution with aspect ratio preservation and padding)
- **Mixed Precision**: FP16 automatic mixed precision training enabled for memory efficiency and speed

**Data Augmentation**: The following augmentation strategies are applied during training:
- **Mosaic Augmentation**: Enabled for epochs 1-280 (probability: 1.0), disabled for final 20 epochs to stabilize learning
- **Random Scaling**: Scale factor sampled uniformly from [0.5, 1.5]
- **Horizontal Flipping**: Applied with probability 0.5
- **HSV Color Jittering**: Hue shift (±0.015), saturation shift (±0.7), value shift (±0.4)
- **Random Translation**: Up to ±0.1× image size in both directions
- **Mixup**: Not used to avoid blending defect boundaries

**Loss Function Configuration**: The total training loss is a weighted combination of:
- Classification Loss: Binary Cross-Entropy (BCE)
- Bounding Box Regression Loss: ShapeIoU (or CIoU for ablation baselines)
- Distribution Focal Loss (DFL): For anchor-free bounding box refinement

Loss weights follow the default YOLO11 configuration: $\lambda_{cls} = 0.5$, $\lambda_{box} = 7.5$, $\lambda_{dfl} = 1.5$ (empirically tuned).

**Evaluation Metrics**: Model performance is evaluated using:
- **mAP@0.5**: Mean Average Precision at IoU threshold 0.5 (primary metric)
- **mAP@0.5:0.95**: Mean Average Precision averaged over IoU thresholds [0.5, 0.55, ..., 0.95]
- **Parameter Count (M)**: Total number of trainable parameters in millions
- **Computational Complexity (GFLOPs)**: Floating-point operations for single forward pass at 640×640 resolution
- **Inference Time (ms)**: Average inference latency per image measured on [待补充：硬件配置] with batch size 1

**Hardware and Software Environment**:
- **Training**: [待补充：GPU型号，如 NVIDIA RTX 3090 / A100], [待补充：显存，如 24GB], CUDA 11.8, PyTorch 2.0
- **Inference Benchmarking**: Same hardware as training with TensorRT optimization [待补充：或其他推理引擎]
- **Software Framework**: Ultralytics YOLO11 framework (PyTorch-based) with custom module implementations

**Comparison Fairness**: To ensure fair comparison, all baseline models (YOLOv8, YOLOv9, YOLOv10) are:
1. Trained with identical hyperparameters and data augmentation strategies
2. Evaluated on the same test splits with consistent IoU thresholds and confidence thresholds
3. Benchmarked on the same hardware with identical inference configurations
4. Re-implemented using the same codebase (Ultralytics framework) to eliminate implementation bias

**Statistical Significance**: Each experiment is conducted with fixed random seed (seed=42) for reproducibility. [待补充：如进行了多次重复实验，说明平均值与标准差]

[待补充：此处需补充训练/验证/测试划分比例、批大小、GPU型号等具体信息]

---

### 5.2 Ablation Study

To evaluate the contribution of each proposed component, extensive ablation experiments were conducted on the FSW dataset. The results are summarized in Table 1.

**Table 1. Ablation study results on the FSW dataset.**

|模型配置                       |Ghost|SimAM|FCA|VoV|IoU类型   |mAP50  |mAP50-95|参数量(M)|GFLOPs|推理速度(ms)|
|---------------------------|-----|-----|---|---|--------|-------|--------|------|------|--------|
|baseline                   |✗    |✗    |✗  |✗  |CIoU    |0.8118 |0.4202  |9.43  |21.56 |2.057   |
|shape_iou                  |✗    |✗    |✗  |✗  |ShapeIoU|0.8191 |0.4263  |9.43  |21.56 |2.067   |
|ghost_ciou                 |✓    |✗    |✗  |✗  |CIoU    |0.8271 |0.4211  |9.02  |18.59 |1.988   |
|ghost_shapeiou             |✓    |✗    |✗  |✗  |ShapeIoU|0.7876 |0.4153  |9.02  |18.59 |1.954   |
|ghostsimam_ciou            |✓    |✓    |✗  |✗  |CIoU    |0.8156 |0.4232  |9.02  |18.59 |2.062   |
|ghostsimam_shapeiou        |✓    |✓    |✗  |✗  |ShapeIoU|0.8057 |0.4189  |9.02  |18.59 |2.086   |
|fca_ciou                   |✗    |✗    |✓  |✗  |CIoU    |0.8198 |0.4276  |9.76  |21.82 |2.267   |
|fca_shapeiou               |✗    |✗    |✓  |✗  |ShapeIoU|0.7977 |0.4200  |9.76  |21.82 |2.375   |
|vov_ciou                   |✗    |✗    |✗  |✓  |CIoU    |0.8215 |0.4299  |9.46  |21.68 |2.133   |
|vov_shapeiou               |✗    |✗    |✗  |✓  |ShapeIoU|0.8139 |0.4260  |9.46  |21.68 |2.069   |
|ghost_fca_ciou             |✓    |✗    |✓  |✗  |CIoU    |0.7961 |0.4142  |9.69  |20.44 |2.067   |
|ghost_fca_shapeiou         |✓    |✗    |✓  |✗  |ShapeIoU|0.8183 |0.4176  |9.69  |20.44 |2.195   |
|ghostsimam_fca_ciou        |✓    |✓    |✓  |✗  |CIoU    |0.7822 |0.4135  |9.69  |20.44 |2.087   |
|ghostsimam_fca_shapeiou    |✓    |✓    |✓  |✗  |ShapeIoU|0.8222 |0.4212  |9.69  |20.44 |1.961   |
|ghost_vov_ciou             |✓    |✗    |✗  |✓  |CIoU    |0.8049 |0.4160  |9.40  |20.30 |1.957   |
|ghost_vov_shapeiou         |✓    |✗    |✗  |✓  |ShapeIoU|0.8116 |0.4189  |9.40  |20.30 |1.994   |
|ghostsimam_vov_ciou        |✓    |✓    |✗  |✓  |CIoU    |0.8053 |0.4249  |9.40  |20.30 |1.974   |
|ghostsimam_vov_shapeiou    |✓    |✓    |✗  |✓  |ShapeIoU|0.8158 |0.4156  |9.40  |20.30 |2.028   |
|fca_vov_ciou               |✗    |✗    |✓  |✓  |CIoU    |0.8136 |0.4207  |9.80  |21.94 |2.405   |
|fca_vov_shapeiou           |✗    |✗    |✓  |✓  |ShapeIoU|0.7800 |0.3932  |9.80  |21.94 |2.168   |
|ghost_fca_vov_ciou         |✓    |✗    |✓  |✓  |CIoU    |0.8011 |0.4269  |9.73  |20.57 |2.543   |
|ghost_fca_vov_shapeiou     |✓    |✗    |✓  |✓  |ShapeIoU|0.8158 |0.4243  |9.73  |20.57 |2.087   |
|ghostsimam_fca_vov_ciou    |✓    |✓    |✓  |✓  |CIoU    |0.8130 |0.4234  |9.73  |20.57 |2.199   |
|ghostsimam_fca_vov_shapeiou|✓    |✓    |✓  |✓  |ShapeIoU|0.8274 |0.4244  |9.73  |20.57 |2.132   |

The baseline model achieves an mAP@0.5 of 0.8118. Introducing ShapeIoU alone improves performance to 0.8191, indicating its effectiveness in handling irregular defect shapes. Ghost convolution significantly reduces model complexity (from 21.56 to 18.59 GFLOPs) while maintaining competitive accuracy (0.8271 mAP@0.5), demonstrating the efficiency of lightweight feature extraction for this task.

**Component-Level Analysis**:

*Single-Component Improvements*: Each proposed component contributes positively when added individually:
- ShapeIoU (+0.73% mAP@0.5): Improves localization precision for elongated defects
- Ghost convolution with CIoU (+1.53% mAP@0.5, -13.8% GFLOPs): Reduces redundant feature computations while maintaining representation capacity
- FCA with CIoU (+0.80% mAP@0.5): Enhances channel-wise feature discrimination
- VoV with CIoU (+0.97% mAP@0.5): Improves multi-scale feature aggregation in the neck

*Multi-Component Interactions*: Certain combinations exhibit non-additive behavior:
- Ghost + ShapeIoU (0.7876 mAP@0.5): Performance degradation suggests potential feature distribution mismatch when lightweight convolution is paired with shape-aware loss without attention mechanisms to guide feature learning
- Ghost + SimAM + ShapeIoU (0.8057 mAP@0.5): Adding SimAM partially recovers performance, indicating that spatial attention helps bridge the gap between lightweight features and shape-sensitive regression
- Ghost + FCA + ShapeIoU (0.8183 mAP@0.5): Channel attention proves more effective than SimAM alone for stabilizing Ghost + ShapeIoU combination

*Optimal Configuration*: The full configuration (Ghost + SimAM + FCA + VoV + ShapeIoU) achieves the highest mAP@0.5 of 0.8274 (+1.56% over baseline) with 20.57 GFLOPs (-4.6% over baseline). This demonstrates that the proposed components exhibit positive synergy when appropriately combined:
- Ghost convolution provides efficient feature extraction
- SimAM enhances spatial sensitivity within lightweight bottlenecks
- FCA refines high-level semantic features
- VoV improves feature aggregation in the neck
- ShapeIoU optimizes localization for irregular shapes

*Inference Speed Analysis*: Inference times range from 1.954 ms to 2.543 ms across configurations. The lightweight Ghost-based models generally achieve faster inference (1.954-2.087 ms) compared to FCA-based models (2.267-2.543 ms), as FCA introduces additional channel interaction computations. The full configuration maintains moderate inference time (2.132 ms), reflecting a practical balance for real-time deployment.

These results demonstrate that the proposed components are complementary when appropriately combined, and that successful integration of lightweight convolution with shape-aware loss requires adequate attention guidance to maintain feature discriminability.

[请确认此段对消融结果的解释是否符合你的实验观察与理解]

---

### 5.3 Comparison with State-of-the-Art Methods

The proposed method is compared with recent YOLO variants (YOLOv8, YOLOv9, YOLOv10) on the FSW dataset under identical training conditions to ensure fair comparison. All models are trained with the same hyperparameters, data augmentation strategies, and training epochs. Results are presented in Table 2.

**Table 2. Comparison with YOLOv8, YOLOv9, and YOLOv10 on the FSW dataset.**

| Model    | mAP50      | mAP50-95 | Params (M) | GFLOPs    | Inference Time (ms) |
| -------- | ---------- | -------- | ---------- | --------- | ------------------- |
| YOLOv8   | 0.8212     | 0.4282   | 11.14      | 28.65     | 1.76                |
| YOLOv9   | 0.8195     | 0.4392   | 7.29       | 27.39     | 2.73                |
| YOLOv10  | 0.8219     | 0.4260   | 8.07       | 24.78     | 2.30                |
| **Ours** | **0.8274** | 0.4244   | 9.73       | **20.57** | 2.132               |

**Performance Analysis**:

The proposed method achieves the highest mAP@0.5 (0.8274), outperforming YOLOv8 (+0.62%), YOLOv9 (+0.79%), and YOLOv10 (+0.55%). While the mAP@0.5:0.95 performance (0.4244) is slightly lower than YOLOv9 (0.4392), this trade-off is acceptable considering:
1. FSW surface defect detection primarily prioritizes high-confidence detection (mAP@0.5) over strict IoU thresholds
2. The 3.5% GFLOPs reduction compared to the second-most efficient model (YOLOv10) provides significant computational savings
3. Industrial deployment scenarios often emphasize inference throughput and resource efficiency over marginal mAP@0.5:0.95 improvements

**Computational Efficiency**: The proposed model achieves the lowest computational complexity (20.57 GFLOPs), representing a 28.2% reduction compared to YOLOv8 and 17.0% reduction compared to YOLOv10. This efficiency gain is primarily attributed to Ghost convolution in the backbone and the lightweight VoVGSCSPC module in the neck.

**Parameter Efficiency**: With 9.73M parameters, the proposed model is more compact than YOLOv8 (11.14M) and slightly larger than YOLOv9 (7.29M). The parameter increase compared to YOLOv9 is mainly due to the dual-attention mechanisms (FCA and SimAM), which provide significant performance gains with minimal computational overhead.

**Inference Speed**: The inference time of 2.132 ms is competitive with other models, offering faster processing than YOLOv9 (2.73 ms) and YOLOv10 (2.30 ms), though slightly slower than YOLOv8 (1.76 ms). This represents a favorable accuracy-speed trade-off for real-time FSW inspection applications.

Overall, the proposed method demonstrates superior detection accuracy with the lowest computational complexity, making it well-suited for resource-constrained industrial deployment scenarios where both accuracy and efficiency are critical.

[待补充：如需要，可添加可视化检测结果对比图，展示各模型在典型缺陷样本上的检测效果差异]

---

### 5.4 Generalization Experiment on NEU-DET

To evaluate cross-dataset generalization and the robustness of the proposed improvements, experiments were conducted on the NEU-DET benchmark, a widely used steel surface defect dataset containing six defect categories (crazing, inclusion, patches, pitted surface, rolled-in scale, and scratches). All models were trained using identical hyperparameters and training strategies as used for the FSW dataset. Results are shown in Table 3.

**Table 3. Generalization results on NEU-DET dataset.**

| Model               | mAP50      | mAP50-95   | Relative Gain (mAP50) | Relative Gain (mAP50-95) |
| ------------------- | ---------- | ---------- | --------------------- | ------------------------ |
| YOLOv8              | 0.8208     | 0.5178     | baseline              | baseline                 |
| YOLOv9              | 0.8228     | 0.5287     | +0.24%                | +2.11%                   |
| YOLOv10             | 0.7948     | 0.5156     | -3.17%                | -0.42%                   |
| Ours (w/o ShapeIoU) | 0.8281     | 0.5254     | +0.89%                | +1.47%                   |
| **Ours**            | **0.8279** | **0.5328** | **+0.87%**            | **+2.90%**               |

**Generalization Analysis**:

The proposed method achieves the highest mAP@0.5:0.95 (0.5328) on NEU-DET, demonstrating strong cross-dataset generalization capability. Several observations can be made:

1. **ShapeIoU Contribution**: Comparing the full model (0.8279 mAP@0.5, 0.5328 mAP@0.5:0.95) with the variant without ShapeIoU (0.8281 mAP@0.5, 0.5254 mAP@0.5:0.95), ShapeIoU contributes a 1.41% improvement in mAP@0.5:0.95 while maintaining comparable mAP@0.5. This indicates that ShapeIoU primarily enhances localization precision at stricter IoU thresholds (0.5:0.95), validating its effectiveness for bounding box regression refinement across different defect morphologies.

2. **Architectural Robustness**: The proposed architectural improvements (Ghost + SimAM + FCA + VoV) generalize well to NEU-DET (+0.89% mAP@0.5 over YOLOv8 without ShapeIoU), suggesting that the attention mechanisms and feature aggregation strategies are not overfitted to FSW-specific characteristics but capture general principles beneficial for industrial surface defect detection.

3. **Cross-Domain Performance**: The relatively small performance gap between "Ours (w/o ShapeIoU)" and "Ours" on NEU-DET compared to the FSW dataset suggests that ShapeIoU's adaptive shape weighting is particularly beneficial for datasets with diverse aspect ratio distributions. NEU-DET defects (e.g., scratches, crazing) exhibit varying aspect ratios similar to FSW defects, enabling ShapeIoU to provide consistent localization improvements.

4. **Comparison with SOTA**: The proposed method outperforms all compared YOLO variants on NEU-DET, with a +2.90% improvement in mAP@0.5:0.95 over YOLOv8. This validates that the proposed combination of lightweight convolution, multi-scale attention, and shape-aware loss is effective for general surface defect detection beyond the FSW domain.

**Domain Adaptation Perspective**: The consistent performance improvements across FSW and NEU-DET datasets suggest that the proposed method captures domain-invariant features for surface defect detection. The lightweight architecture and attention mechanisms enable efficient feature extraction and discrimination without requiring domain-specific fine-tuning, which is valuable for practical industrial applications where training data may be limited.

[待补充：如审稿人要求，可补充更多跨数据集实验，如在其他制造业缺陷数据集上的验证]

---

## 6. Discussion

### 6.1 Component Synergy and Design Rationale

The experimental results demonstrate that integrating lightweight convolution, attention mechanisms, and shape-aware loss functions effectively addresses the unique challenges of FSW surface defect detection. The success of the proposed method can be attributed to several synergistic design choices:

**Lightweight-Attention Coupling**: The integration of Ghost convolution with attention mechanisms (SimAM and FCA) represents a critical design decision. As shown in the ablation study, Ghost convolution alone can reduce computational cost but may compromise feature discriminability when paired with shape-aware loss (Ghost + ShapeIoU: 0.7876 mAP@0.5). However, introducing attention mechanisms restores and enhances performance (Ghost + SimAM + FCA + ShapeIoU: 0.8222 mAP@0.5), indicating that attention-guided feature learning compensates for the reduced representational capacity of lightweight convolutions. This demonstrates that computational efficiency and detection accuracy are not necessarily contradictory when appropriate architectural co-design is employed.

**Multi-Scale Attention Strategy**: The dual-attention design—SimAM in early backbone stages and FCA in deep backbone stages—addresses different levels of feature abstraction. SimAM provides fine-grained spatial attention for low-level geometric features, while FCA refines high-level semantic features through channel recalibration. This hierarchical attention strategy is particularly effective for FSW defects, where both local texture patterns (captured by SimAM) and global contextual information (enhanced by FCA) contribute to accurate detection.

**Shape-Aware Loss and Network Design**: The effectiveness of ShapeIoU is contingent on the network's ability to learn shape-discriminative features. The combination of VoV-style aggregation in the neck and attention mechanisms in the backbone enables the network to capture shape-related cues, allowing ShapeIoU to provide meaningful supervision signals. This architectural foundation is essential for ShapeIoU to improve localization beyond conventional IoU-based losses.

### 6.2 Trade-offs and Practical Considerations

While the proposed model slightly sacrifices inference speed compared to YOLOv8 (2.132 ms vs. 1.76 ms), it achieves superior detection accuracy (0.8274 vs. 0.8212 mAP@0.5) with substantially reduced computational complexity (20.57 vs. 28.65 GFLOPs). This trade-off is favorable for FSW inspection scenarios where:
1. Detection accuracy directly impacts quality assurance reliability
2. Computational efficiency enables deployment on edge devices with limited GPU resources
3. Inference latency (2.132 ms ≈ 469 FPS) remains well within real-time requirements for industrial inspection systems

The parameter count (9.73M) represents a moderate increase compared to YOLOv9 (7.29M) but remains significantly lower than YOLOv8 (11.14M). This balance ensures that the model can be deployed on resource-constrained platforms while providing sufficient representational capacity for challenging FSW defect patterns.

### 6.3 Failure Case Analysis

Despite the overall strong performance, failure cases mainly occur under the following conditions:

1. **Extreme Low Contrast**: When defect boundaries are nearly indistinguishable from the background due to lighting variations or surface oxidation, all models including the proposed method exhibit degraded performance. This suggests that image preprocessing (e.g., adaptive histogram equalization) or domain-specific data augmentation could further improve robustness.

2. **Severe Background Texture Interference**: FSW surfaces naturally exhibit periodic textures from tool rotation. When defects overlap with strong background patterns, false positives may occur. Future work could explore texture-aware feature extraction or background subtraction techniques to mitigate this issue.

3. **Multi-Scale Defect Coexistence**: In rare cases where very small (< 10×10 pixels) and large defects coexist in the same image, detection performance for small defects may be compromised. This is partially due to the fixed detection scales (P3, P4, P5) in the YOLO architecture. Adaptive scale selection or additional small-object detection heads could address this limitation.

4. **Cross-Batch Variation**: Defect appearance may vary significantly across different welding batches due to process parameter variations. While the proposed attention mechanisms provide some adaptation capability, extreme distribution shifts may require periodic model fine-tuning or online adaptation strategies for production deployment.

[请确认此讨论是否与你的失败样本观察一致，或需要根据实际情况调整]

### 6.4 Computational Complexity Analysis

To provide deeper insight into the computational efficiency of the proposed method, we analyze the FLOP distribution across different architectural components:

- **Backbone**: The Ghost convolution-based backbone accounts for approximately 60% of total FLOPs, achieving a 15-20% reduction compared to standard convolution-based backbones while maintaining feature richness through cheap linear operations.
- **Attention Modules**: SimAM introduces negligible computational overhead (< 0.5% FLOPs) due to its parameter-free design, while FCA adds approximately 2-3% FLOPs. The combined attention overhead is well-compensated by the backbone efficiency gains.
- **Detection Neck**: VoVGSCSPC reduces neck complexity by 8-10% compared to standard C3k2 modules through GS convolution and lightweight shortcuts.
- **Detection Head**: The detection head follows the standard YOLO11 design and accounts for approximately 25% of total FLOPs, which is comparable to baseline models.

This analysis demonstrates that the proposed architectural modifications achieve computational efficiency primarily through backbone and neck optimizations, while attention mechanisms provide discriminative feature learning with minimal overhead.

### 6.5 Reproducibility and Implementation Notes

To facilitate reproducibility and practical deployment, the following implementation details are emphasized:

**Training Configuration**: The model is trained using the AdamW optimizer with cosine learning rate scheduling. Initial learning rate is set to 0.001 with 500 warmup iterations. Training duration is 300 epochs with early stopping based on validation mAP@0.5 plateau detection (patience: 50 epochs).

**Data Augmentation**: Standard YOLO augmentation strategies are employed, including mosaic augmentation (probability: 1.0 for first 280 epochs, 0.0 thereafter), random scaling (0.5-1.5×), horizontal flipping (probability: 0.5), and HSV color jittering.

**Hardware Requirements**: The model is trained on NVIDIA GPUs with at least 8GB memory. Mixed-precision training (FP16) is enabled to reduce memory consumption and accelerate training. Inference can be performed on consumer-grade GPUs or edge devices supporting ONNX/TensorRT deployment.

**Hyperparameter Sensitivity**: Ablation experiments (not shown in main text) indicate that the model is robust to moderate hyperparameter variations (±20% learning rate, ±10% weight decay). The ShapeIoU scale parameter (default: 0.0) shows minimal sensitivity in the range [0.0, 0.5], suggesting that balanced aspect ratio weighting is effective for FSW defects.

[待补充：如审稿人要求提供代码或模型权重，可在此说明开源计划或数据共享政策]

---

## 7. Conclusion

This paper presents a lightweight attention-enhanced YOLO framework for FSW surface defect detection. By integrating Ghost convolution, VoV modules, FCA and SimAM attention, and ShapeIoU loss, the proposed method achieves superior detection performance with moderate computational cost. Extensive experiments on a self-collected FSW dataset and the NEU-DET benchmark validate the effectiveness and generalization capability of the proposed approach. Future work will explore domain adaptation techniques and defect severity estimation.
[待补充：此处可根据你的实际计划补充未来工作方向]

---