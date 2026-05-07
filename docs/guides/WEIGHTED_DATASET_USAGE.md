# Using YOLOWeightedDataset to Handle Class Imbalance

This document provides step-by-step instructions for using the `YOLOWeightedDataset` class to address data imbalance in YOLO training.

## Overview

The `YOLOWeightedDataset` is a custom dataset class that extends the standard `YOLODataset` to provide weighted sampling based on class frequencies. This helps to balance the distribution of classes during training when dealing with imbalanced datasets.

## How It Works

1. **Class Counting**: The dataset counts the number of instances for each class in the training data.
2. **Weight Calculation**: Each class is assigned a weight inversely proportional to its frequency (rare classes get higher weights).
3. **Probability Distribution**: Each image receives a sampling probability based on the aggregated weights of its objects.
4. **Weighted Sampling**: During training, images are sampled according to these probabilities, ensuring rare classes are seen more frequently.

## Usage

### Method 1: Direct Instantiation (Recommended for Testing)

```python
from ultralytics import YOLO
from ultralytics.data.weighted_dataset import YOLOWeightedDataset

# Create a weighted dataset directly
dataset = YOLOWeightedDataset(
    img_path="path/to/train/images",
    data={"names": {0: "class1", 1: "class2", 2: "class3"}},
    imgsz=640,
    augment=True,
)

# Check class distribution
print(f"Class counts: {dataset.counts}")
print(f"Class weights: {dataset.class_weights}")
```

### Method 2: Monkey-Patching (For Training Integration)

To integrate the weighted dataset into the YOLO training pipeline, use monkey-patching:

```python
from ultralytics import YOLO
import ultralytics.data.build as build
from ultralytics.data.weighted_dataset import YOLOWeightedDataset

# Replace the default YOLODataset with YOLOWeightedDataset
build.YOLODataset = YOLOWeightedDataset

# Now train as usual - the weighted dataset will be used automatically
model = YOLO("yolo11n.pt")
results = model.train(
    data="your_dataset.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
)
```

### Method 3: Using in Custom Training Script

```python
from ultralytics.data.weighted_dataset import YOLOWeightedDataset
from ultralytics.data.build import build_dataloader
import ultralytics.data.build as build

# Monkey-patch before building dataloaders
build.YOLODataset = YOLOWeightedDataset

# Your custom training code here
# The weighted dataset will be used when build_yolo_dataset is called
```

## Configuration Options

### Custom Aggregation Function

By default, the dataset uses `np.mean` to aggregate weights for images with multiple objects. You can customize this:

```python
import numpy as np
from ultralytics.data.weighted_dataset import YOLOWeightedDataset

# Create dataset
dataset = YOLOWeightedDataset(
    img_path="path/to/train/images",
    data={"names": {0: "class1", 1: "class2"}},
)

# Change aggregation function after initialization
dataset.agg_func = np.max  # Use maximum weight instead of mean
dataset.weights = dataset.calculate_weights()  # Recalculate
dataset.probabilities = dataset.calculate_probabilities()  # Recalculate
```

Available aggregation functions:
- `np.mean` (default): Average weight of all objects in the image
- `np.max`: Maximum weight among all objects
- `np.min`: Minimum weight among all objects
- `np.sum`: Sum of all object weights
- Custom function: Any function that takes an array and returns a scalar

### Custom Class Weights

Instead of using automatic inverse frequency weighting, you can specify custom weights:

```python
from ultralytics.data.weighted_dataset import YOLOWeightedDataset
import numpy as np

# Create dataset
dataset = YOLOWeightedDataset(
    img_path="path/to/train/images",
    data={"names": {0: "class1", 1: "class2", 2: "class3"}},
)

# Override with custom weights
# Higher values = more sampling priority
dataset.class_weights = np.array([1.0, 2.0, 5.0])  # Give class3 highest priority
dataset.weights = dataset.calculate_weights()
dataset.probabilities = dataset.calculate_probabilities()
```

## Understanding the Weighting Strategy

### Default Weighting (Inverse Frequency)

For a dataset with class distribution:
- Class A: 1000 instances
- Class B: 500 instances  
- Class C: 100 instances

The weights would be:
- Class A weight: (1000 + 500 + 100) / 1000 = 1.6
- Class B weight: (1000 + 500 + 100) / 500 = 3.2
- Class C weight: (1000 + 500 + 100) / 100 = 16.0

This means images containing Class C will be sampled ~10x more often than images with only Class A.

### Image-Level Weighting

For an image containing multiple objects (e.g., 2 from Class A, 1 from Class C):
- Object weights: [1.6, 1.6, 16.0]
- Image weight (with `np.mean`): (1.6 + 1.6 + 16.0) / 3 = 6.4

## Validation Behavior

During validation (when `prefix` does not contain "train"), the weighted sampling is **disabled** and images are accessed sequentially. This ensures:
- Deterministic validation results
- All validation images are evaluated
- No bias in validation metrics

## Complete Training Example

```python
#!/usr/bin/env python
"""
Example training script with weighted dataset for class imbalance.
"""

from ultralytics import YOLO
import ultralytics.data.build as build
from ultralytics.data.weighted_dataset import YOLOWeightedDataset

def main():
    # Step 1: Apply monkey-patch to use weighted dataset
    build.YOLODataset = YOLOWeightedDataset
    
    # Step 2: Load model
    model = YOLO("yolo11n.pt")  # or any other YOLO model
    
    # Step 3: Train with your imbalanced dataset
    results = model.train(
        data="imbalanced_dataset.yaml",  # your dataset config
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,  # GPU device
        project="runs/weighted_training",
        name="exp1",
    )
    
    print(f"Training complete. Results: {results}")

if __name__ == "__main__":
    main()
```

## Verifying the Implementation

To verify that weighted sampling is working:

```python
from ultralytics.data.weighted_dataset import YOLOWeightedDataset
import numpy as np

# Create dataset
dataset = YOLOWeightedDataset(
    img_path="path/to/train/images",
    data={"names": {0: "class1", 1: "class2"}},
)

# Print statistics
print(f"Total images: {len(dataset.labels)}")
print(f"Class counts: {dataset.counts}")
print(f"Class weights: {dataset.class_weights}")
print(f"Image weights (first 5): {dataset.weights[:5]}")
print(f"Image probabilities (first 5): {dataset.probabilities[:5]}")
print(f"Sum of probabilities: {sum(dataset.probabilities):.4f}")  # Should be ~1.0

# Sample some images and count class distribution
sample_size = 1000
sampled_classes = []
for _ in range(sample_size):
    idx = np.random.choice(len(dataset.labels), p=dataset.probabilities)
    label = dataset.labels[idx]
    cls = label["cls"].reshape(-1).astype(int)
    sampled_classes.extend(cls.tolist())

unique, counts = np.unique(sampled_classes, return_counts=True)
print(f"\nSampled class distribution ({sample_size} images):")
for c, count in zip(unique, counts):
    print(f"  Class {c}: {count} instances")
```

## Limitations and Considerations

1. **Training Mode Only**: Weighted sampling only applies during training. Validation uses sequential access.

2. **Memory**: The dataset stores weights and probabilities for all images, adding minimal memory overhead.

3. **Determinism**: Weighted sampling introduces randomness. For deterministic training, use the standard YOLODataset.

4. **Multi-Object Images**: Images with multiple objects use an aggregation function (default: mean). This may not be optimal for all use cases.

5. **Empty Images**: Background images (no objects) receive a default weight of 1.0.

## Troubleshooting

### Issue: "No labels found" error

**Solution**: Ensure your dataset has valid labels and the path is correct.

### Issue: Weighted sampling not working

**Solution**: Check that `prefix` contains "train" - this determines if training mode is active.

### Issue: Custom weights not applied

**Solution**: After setting custom weights, remember to recalculate:
```python
dataset.weights = dataset.calculate_weights()
dataset.probabilities = dataset.calculate_probabilities()
```

## Additional Resources

- [Ultralytics YOLO Documentation](https://docs.ultralytics.com)
- [Handling Imbalanced Datasets](https://docs.ultralytics.com/guides/preprocessing_annotated_data/)
- [Custom Training](https://docs.ultralytics.com/modes/train/)
