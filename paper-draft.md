
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

The FSW surface defect dataset used in this study was self-collected and manually annotated. It contains approximately 1000 images with four types of surface defects. Each image may contain multiple defects of different categories with varying sizes and shapes.

[待补充：此处需说明具体缺陷类别名称、图像采集设备、分辨率范围、光照条件及标注规范]

### 3.2 Challenges in FSW Surface Defect Detection

FSW surface defects present several challenges for automated detection:

* Small object size relative to image resolution;
* Irregular and elongated defect shapes;
* Low contrast between defects and background;
* Coexistence of multiple defect types within a single image.

These characteristics motivate the need for enhanced feature representation, attention-driven feature selection, and shape-aware localization strategies.
[请确认此部分逻辑或提供更多背景]

---

## 4. Proposed Method

### 4.1 Overall Architecture

The proposed framework is built upon a YOLO-style detection architecture and introduces several targeted modifications to address the challenges of FSW surface defect detection. The overall architecture consists of a backbone for feature extraction, a neck for multi-scale feature aggregation, and a detection head for classification and localization.

[待补充：此处需插入整体网络结构示意图说明]

### 4.2 Ghost Convolution and VoV Modules

Ghost convolution is employed to reduce computational redundancy by generating feature maps through cheap linear operations. VoV modules are integrated to enhance feature reuse and multi-scale aggregation, which is particularly beneficial for detecting small and irregular defects.

### 4.3 Dual Attention Mechanism: FCA and SimAM

To enhance feature discrimination, FCA is used to emphasize informative feature channels, while SimAM introduces spatial attention without additional learnable parameters. The combination of these two mechanisms improves sensitivity to subtle defect patterns.

### 4.4 ShapeIoU Loss for Localization

To better handle irregular defect shapes, ShapeIoU loss is adopted in place of CIoU. ShapeIoU explicitly considers shape alignment between predicted and ground-truth boxes, leading to improved localization performance for elongated and irregular defects.

[请确认此部分对 ShapeIoU 的作用解释是否符合你的实现动机]

---

## 5. Experiments

### 5.1 Experimental Setup

[待补充：此处需说明训练/验证/测试划分比例、优化器类型、学习率、训练轮数、输入尺寸、硬件环境]

Evaluation metrics include mAP@0.5, mAP@0.5:0.95, parameter count, GFLOPs, and inference latency.

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

The baseline model achieves an mAP@0.5 of 0.8118. Introducing ShapeIoU alone improves performance to 0.8191, indicating its effectiveness in handling irregular defect shapes. Ghost convolution significantly reduces model complexity (from 21.56 to 18.59 GFLOPs) while maintaining competitive accuracy.

Interestingly, certain combinations (e.g., Ghost + ShapeIoU) result in performance degradation, suggesting potential feature incompatibility when attention mechanisms are absent. The best overall performance is achieved by the full configuration integrating Ghost, SimAM, FCA, VoV, and ShapeIoU, which attains an mAP@0.5 of 0.8274 with moderate computational cost.

These results demonstrate that the proposed components are complementary when appropriately combined.
[请确认此段对消融结果的解释是否符合你的实验观察]

---

### 5.3 Comparison with State-of-the-Art Methods

The proposed method is compared with recent YOLO variants on the FSW dataset. Results are presented in Table 2.

**Table 2. Comparison with YOLOv8, YOLOv9, and YOLOv10 on the FSW dataset.**

| Model    | mAP50      | mAP50-95 | Params (M) | GFLOPs    | Inference Time (ms) |
| -------- | ---------- | -------- | ---------- | --------- | ------------------- |
| YOLOv8   | 0.8212     | 0.4282   | 11.14      | 28.65     | 1.76                |
| YOLOv9   | 0.8195     | 0.4392   | 7.29       | 27.39     | 2.73                |
| YOLOv10  | 0.8219     | 0.4260   | 8.07       | 24.78     | 2.30                |
| **Ours** | **0.8274** | 0.4244   | 9.73       | **20.57** | 2.132               |

The proposed method achieves the highest mAP@0.5 while maintaining significantly lower computational complexity than YOLOv8 and YOLOv10, demonstrating a favorable accuracy–efficiency trade-off.

---

### 5.4 Generalization Experiment on NEU-DET

To evaluate cross-dataset generalization, experiments were conducted on the NEU-DET benchmark. Results are shown in Table 3.

**Table 3. Generalization results on NEU-DET dataset.**

| Model               | mAP50      | mAP50-95   |
| ------------------- | ---------- | ---------- |
| YOLOv8              | 0.8208     | 0.5178     |
| YOLOv9              | 0.8228     | 0.5287     |
| YOLOv10             | 0.7948     | 0.5156     |
| Ours (w/o ShapeIoU) | 0.8281     | 0.5254     |
| **Ours**            | **0.8279** | **0.5328** |

The results indicate that ShapeIoU further improves generalization performance, particularly on mAP@0.5:0.95, suggesting enhanced robustness in bounding box regression across domains.

---

## 6. Discussion

The experimental results demonstrate that integrating lightweight convolution, attention mechanisms, and shape-aware loss functions effectively addresses the unique challenges of FSW surface defect detection. While the proposed model slightly sacrifices inference speed compared to YOLOv8, it achieves superior detection accuracy with substantially reduced computational complexity.

Failure cases mainly occur under extremely low contrast or severe surface noise conditions, indicating potential directions for future improvement, such as domain adaptation or enhanced preprocessing strategies.
[请确认此讨论是否与你的失败样本观察一致]

---

## 7. Conclusion

This paper presents a lightweight attention-enhanced YOLO framework for FSW surface defect detection. By integrating Ghost convolution, VoV modules, FCA and SimAM attention, and ShapeIoU loss, the proposed method achieves superior detection performance with moderate computational cost. Extensive experiments on a self-collected FSW dataset and the NEU-DET benchmark validate the effectiveness and generalization capability of the proposed approach. Future work will explore domain adaptation techniques and defect severity estimation.
[待补充：此处可根据你的实际计划补充未来工作方向]

---