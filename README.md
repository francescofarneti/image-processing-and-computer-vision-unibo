# assignments a.y. 2023-2024

# 🛒 Computer Vision assignment 1 — Product Detection on Store Shelves
 
## Overview
 
Computer vision-based object detection techniques can be applied in supermarket settings to build a system capable of identifying products on store shelves. This system could be used to:
 
- Assist **visually impaired customers** in navigating the store;
- Automate common **store management tasks**, such as detecting low-stock or misplaced products.
Given a reference image for each product, the system must identify such products from a single picture of a store shelf.
 
---
 
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
 
## Deliverables
 
- [ ] Source code of the detection system;
- [ ] Brief report describing the chosen approach and methodology;
- [ ] Output results for all scenes (Tracks A and B) in the specified format;
- [ ] (Optional) Visualizations of detections overlaid on scene images.
---
 
## Notes
 
- The coordinate system origin `(0, 0)` is at the **top-left corner** of the image;
- Bounding box position is reported as the **center** of the box `(cx, cy)`;
- Width and height refer to the bounding box in **pixels**.
 
