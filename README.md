# assignments a.y. 2023-2024

# 🛒 Computer Vision assignment 1 — Product Detection on Store Shelves
 
## Task Description
 
Develop a computer vision system that, given:
- A **reference image** for each product (template/model);
- A **scene image** (a photograph of a store shelf);
is able to **locate and identify** each product within the shelf image.
 
### Required Output
 
For each type of product displayed in the shelf, the system must report:
 
| Field | Description |
|---|---|
| **Number of instances** | How many times that product appears in the scene |
| **Dimensions** | Width and height (in pixels) of the bounding box enclosing each instance |
| **Position** | Center coordinates `(x, y)` of the bounding box in the image reference system |
 
### Example Output
 
```
Product 0 - 2 instances found:
  Instance 1 { position: (256, 328), width: 57px, height: 80px }
  Instance 2 { position: (311, 328), width: 57px, height: 80px }
 
Product 1 - 1 instance found:
  Instance 1 { position: (134, 210), width: 62px, height: 85px }
 
...
```
 
---
 
## Tracks
 
### 🔵 Track A — Single Instance Detection
 
Develop an object detection system to identify a **single instance** of each product, given:
- One reference image per item;
- A scene image.
The system must correctly identify **all products** present in the shelf image.
 
**Data:**
- Models: `ref1.png` → `ref14.png`
- Scenes: `scene1.png` → `scene5.png`
> ⚠️ Scene images are corrupted by noise.
 
---
 
### 🟠 Track B — Multiple Instances Detection
 
Extend the system developed in Track A to also detect **multiple instances** of the same product within a single scene.
 
**Data:**
- Models: `ref15.png` → `ref27.png`
- Scenes: `scene6.png` → `scene12.png`
> ⚠️ Scene images are corrupted by noise.
 
---
 
## Data Structure
 
```
project/
├── Models/
│   ├── ref1.png
│   ├── ref2.png
│   ├── ...
│   ├── ref14.png       ← Track A
│   ├── ref15.png
│   └── ...
│   └── ref27.png       ← Track B
│
└── Scenes/
    ├── scene1.png
    ├── ...
    ├── scene5.png      ← Track A
    ├── scene6.png
    └── ...
    └── scene12.png     ← Track B
```
 
---
 
## Suggested Approach
 
### Preprocessing
- Apply noise reduction (e.g., Gaussian blur, median filter) to scene images to mitigate corruption;
- Normalize brightness and contrast between reference and scene images.
### Detection Methods
Some applicable techniques:
 
- **Template Matching** — classic sliding-window correlation (suitable for Track A);
- **Feature-based Matching** — SIFT, ORB, AKAZE with homography estimation (robust to scale/rotation changes);
- **Deep Learning** — YOLO, Faster R-CNN, or few-shot detection models (suitable for Track B with multiple instances).
### Post-processing
- Apply **Non-Maximum Suppression (NMS)** to eliminate overlapping detections;
- Filter detections by a confidence/score threshold.
---
 
## Evaluation Criteria
 
The system will be evaluated based on:
 
- ✅ Correct identification of all products (detection accuracy);
- ✅ Precision of bounding box localization (IoU with ground truth);
- ✅ Ability to handle multiple instances of the same product (Track B);
- ✅ Robustness to image noise present in the scene images.
---
 
## Notes
 
- The coordinate system origin `(0, 0)` is at the **top-left corner** of the image;
- Bounding box position is reported as the **center** of the box `(cx, cy)`;
- Width and height refer to the bounding box in **pixels**.
 
# 🛒 Assignment 2 — Product Classification with Neural Networks

## 📌 Overview

This notebook addresses the task of **image classification of grocery store products** captured with a smartphone camera. The dataset contains 43 product categories (fruits, vegetables, dairy products, and juices), and the goal is to correctly classify each image into its corresponding class.

The work is split into two parts:

1. **Custom CNN from scratch** — design, train, and justify a convolutional neural network built using PyTorch primitives;
2. **Fine-tuning ResNet-18** — adapt a pretrained ResNet-18 (ImageNet-1K V1) to the grocery dataset, with progressive hyperparameter tuning.

---

## 📂 Dataset

The dataset used is the [GroceryStoreDataset](https://github.com/marcusklasson/GroceryStoreDataset), cloned directly from GitHub. It contains natural smartphone images of grocery products organized in **43 fine-grained classes** across three macro-categories:

| Macro-category | Examples |
|---|---|
| 🍎 Fruits | Apple, Banana, Mango, Pineapple, ... |
| 🥛 Dairy & Drinks | Milk, Yoghurt, Oat-Milk, Juice, ... |
| 🥦 Vegetables | Carrot, Pepper, Zucchini, Garlic, ... |

The dataset is pre-split into **train**, **val**, and **test** sets. A custom `torch.utils.data.Dataset` class is provided and used as the foundation for all data loading pipelines.

---

## 🧠 Part 1 — Custom Convolutional Neural Network

### What's inside

The notebook documents a **step-by-step experimental process** to build a CNN from scratch using PyTorch layers (`nn.Conv2d`, `nn.Linear`, `nn.BatchNorm2d`, etc.) — without using any off-the-shelf `torchvision` model.

### Approach

Rather than jumping to the final model, the notebook follows an **incremental design methodology**:

- Start from a minimal baseline (a simple stack of conv + pool + fc layers);
- Add complexity step by step — deeper architecture, batch normalization, dropout, data augmentation;
- Each addition is **justified** by comparing validation accuracy before and after the change, shown through training/validation loss and accuracy plots.

### Key design choices explored

| Component | Justification |
|---|---|
| Convolutional blocks | Spatial feature extraction with increasing depth |
| Batch Normalization | Stabilizes training, reduces sensitivity to learning rate |
| Dropout | Regularization to counter overfitting |
| Data Augmentation | Random flips, crops, color jitter to improve generalization |
| Optimizer & LR | Tuned through empirical observation of training curves |

### Result

The final custom model achieves a validation accuracy of **~60%** on the GroceryStoreDataset, meeting the target defined in the assignment.

---

## 🔁 Part 2 — Fine-tuning ResNet-18

### What's inside

This section fine-tunes the **PyTorch pretrained ResNet-18 (ImageNet-1K V1)** on the grocery dataset, in two stages:

#### Stage 1 — Same hyperparameters as Part 1
The ResNet-18 is fine-tuned using the same training configuration (optimizer, learning rate, batch size, epochs) used for the best custom model in Part 1. This provides a **direct comparison** between a model trained from scratch and a pretrained one under identical conditions.

#### Stage 2 — Hyperparameter tuning
Building on Stage 1, training hyperparameters are progressively adjusted to push accuracy higher. Choices are justified by:
- Analysis of training/validation curves (overfitting, underfitting, learning rate dynamics);
- References to best practices from literature and established sources.

### Result

After tuning, the fine-tuned ResNet-18 reaches a validation accuracy in the **80–90% range**, as targeted by the assignment.

---

## 📊 Results Summary

| Model | Validation Accuracy |
|---|---|
| Custom CNN (baseline) | ~30–40% |
| Custom CNN (final) | ~60% |
| ResNet-18 (same HPs as Part 1) | ~70–75% |
| ResNet-18 (tuned) | **~80–90%** |

> Exact values are reported in the notebook with accompanying plots.

---

## 🗂️ Repository Structure

```
.
├── notebook.ipynb          # Main notebook with all experiments
├── README.md               # This file
└── GroceryStoreDataset/    # Cloned automatically by the notebook
    ├── train/
    ├── val/
    └── test/
```

---

## ▶️ How to Run

1. Clone this repository;
2. Open `notebook.ipynb` in Jupyter or Google Colab;
3. Run the first cell to clone the GroceryStoreDataset automatically:
   ```bash
   !git clone https://github.com/marcusklasson/GroceryStoreDataset.git
   ```
4. Execute the remaining cells in order.

**Requirements:** `torch`, `torchvision`, `Pillow`, `matplotlib`, `numpy`

---

## 📝 Notes

- All models are implemented in **PyTorch**;
- Part 1 uses only `torch.nn` primitives — no `torchvision.models`;
- Part 2 uses `torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)` as required;
- Every architectural and hyperparameter choice is experimentally motivated inside the notebook.
