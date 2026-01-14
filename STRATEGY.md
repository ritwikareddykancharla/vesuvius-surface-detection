# 🌋 Vesuvius Challenge: Strategy (Target: 0.6+ LB)

This document details the technical strategy implemented in `vesuvius-winning-strategy.ipynb`. This approach is designed to outperform the Host Baseline (0.562) by specifically targeting topological metrics.

## 1. The Core Problem: Topology vs. Accuracy
Standard segmentation models (like plain U-Net) optimize for **pixel accuracy**.
- **Scenario**: A predicted sheet has a tiny 1-pixel hole in the middle.
- **Dice Score**: ~0.99 (Excellent).
- **TopoScore**: Near 0.0 (Terrible, because you created a topological "handle" or "tunnel").

**The Strategy**: We must force the model to prioritize **connectivity** over raw pixel accuracy.

---

## 2. Architecture: Res-UNet 3D
We use a **Residual 3D U-Net** instead of a standard U-Net.

### Why?
- **Vanishing Gradients**: 3D volumes require deep networks to capture context. Standard deep U-Nets often suffer from vanishing gradients. Residual connections (skip connections within blocks) solve this.
- **Instance Normalization**: We use `InstanceNorm3d` instead of `BatchNorm3d`.
    - Batch Norm requires large batch sizes (e.g., 16, 32) to estimate statistics accurately.
    - In 3D DL, we can often only fit Batch Size = 2 or 4. Batch Norm performs poorly here. Instance Norm is independent of batch size.

### Configuration
- **Patch Size**: `128x128x128` (or `192x192x192` if VRAM allows). Larger context is critical for following scrolling sheets.
- **Encoder**: 4 levels of downsampling (Max Pool).
- **Decoder**: Transpose Convolutions for upsampling.

---

## 3. Loss Function: The Secret Weapon (`clDice`)
We use a composite loss function:
$$ Loss = 0.5 \times BCE + 0.5 \times clDice $$

### What is `clDice`?
**Centerline Dice (clDice)** is a topology-aware loss.
1.  **Soft Skeletonization**: It mathematically calculates a "soft skeleton" (medial axis) of both the Prediction and the Ground Truth in a differentiable way.
2.  **Matching**: It measures the overlap between:
    - The **Skeleton of the Prediction** and the **Mask of the Ground Truth** (Precision).
    - The **Skeleton of the Ground Truth** and the **Mask of the Prediction** (Sensitivity).

**Effect**: If the model predicts a broken sheet (a hole), the Ground Truth skeleton will pass through that hole, detecting zero overlap in that region. The loss essentially screams: **"There should be a connection here!"**

---

## 4. Training Pipeline
- **Data Augmentation**:
    - **Random Z-Flip**: The scroll has no "up" or "down" preference locally.
    - **Random XY-Flip**: Symmetry invariance.
    - **Random Intensity Shift**: To handle varying contrast in CT scans.
- **Sampling**:
    - We sample random crops from the volume.
    - *Advanced Tip*: Implement "hard example mining" by sampling more often from areas with surface labels rather than empty background.

---

## 5. Inference & Post-Processing
Winning margins are made here.

### A. TTA (Test Time Augmentation)
During inference, we don't just predict `Model(X)`. We predict 8 versions and average them:
1.  Original
2.  Flip Z
3.  Flip Y
4.  Flip X
5.  Flip Z + Flip Y
6.  ...etc.

**Gain**: Typically **+0.005 to +0.010 LB**.

### B. Frangi Filtering (Hessian-based)
Before thresholding the probability map (0.0 to 1.0), we apply a **Frangi Vesselness filter** (tuned for plates/sheets).
- This filter computes the **Hessian matrix** (second-order derivatives) at every pixel.
- It looks for regions where the curvature is flat in two directions and high in one direction (a sheet).
- It suppresses "blobs" and enhances "sheets".

**Gain**: Crucial for **VOI Score**.

---

## How to Run
Since the notebook editor failed, you can implement this by running the code as a Python script or retrying the notebook. The key dependency is `grad-cam` or `monai` if you want pre-built versions, but the notebook provided has a self-contained implementation of `clDice`.
