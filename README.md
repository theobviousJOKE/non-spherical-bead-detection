# Bead Detection and Orientation Estimation using YOLOv8 + PCA

This project automates the process of detecting **non-spherical beads** in video frames and estimating their **orientation angles** using a hybrid approach combining deep learning and traditional computer vision. The detection is powered by **YOLOv8**, while orientation estimation is handled via **Principal Component Analysis (PCA)**.

Developed under the mentorship of **Prof. Anshu Anand**, this project is part of an academic initiative focused on computer vision applications in object geometry analysis.

---

## Objectives

-  Detect non-spherical beads from high-resolution video data.
-  Estimate the **angular orientation** of beads using PCA on image contours.
-  Generate annotated videos with bounding boxes and angle overlays.
-  Store bounding box and angle data for downstream analysis.

---

##  Tech Stack

| Tool            | Purpose                         |
|-----------------|---------------------------------|
| **Python**      | Programming language            |
| **YOLOv8 (Ultralytics)** | Object detection       |
| **OpenCV**      | Frame extraction & drawing      |
| **scikit-learn (PCA)** | Orientation calculation |
| **Matplotlib / Seaborn** | Visualization         |
| **Torch / torchvision** | Backend for YOLO        |

---

##  Project Structure

```bash
beads_detection/
│
├── dataset_red&trans/               # Dataset directory (YOLO format)
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   └── labels/
│       ├── train/
│       └── val/
│
├── videos/                          # Raw input videos
│   └── fr24_start30sec.mp4
│
├── output/                          # Annotated output videos
│   └── fr24_start30sec_annotated.mp4
│
├── data3.yaml                       # Custom YOLO data config
├── beads_detection.ipynb           # Main notebook
├── yolov8s.pt                       # YOLOv8 small weights
├── requirements.txt                 # Dependencies
└── README.md                        # Documentation
```

##  Setup Instructions

### 1 Clone the Repository

```bash
git clone https://github.com/your-username/beads-detection.git
cd beads-detection
```
### 2 Install Dependencies
```bash
pip install -r requirements.txt
```
or install manually:
```bash
pip install ultralytics opencv-python scikit-learn matplotlib seaborn torch torchvision pyyaml
```

### 3 Prepare the Dataset
dataset_red&trans/
├── images/
│ ├── train/
│ └── val/
└── labels/
├── train/
└── val/


### Directory Structure Details:
- `images/train/` — Contains training images.
- `images/val/` — Contains validation images.
- `labels/train/` — Contains label files corresponding to training images.
- `labels/val/` — Contains label files corresponding to validation images.

### Label File Format:
Each label file should contain one or more lines, each describing an object in the image. The format for each line is:


- `<class_id>`: Integer representing the object class (e.g., `0` for non-spherical bead).
- `<x_center>` and `<y_center>`: Normalized coordinates (values between 0 and 1) of the bounding box center.
- `<width>` and `<height>`: Normalized width and height of the bounding box (values between 0 and 1).

### Example:
For a non-spherical bead labeled as class `0`, a line in the label file could look like:


This line means:
- Class `0` (non-spherical bead)
- Bounding box center at 54.3% width and 28.7% height of the image
- Bounding box width 11.2% and height 6.5% of the image size

Make sure that every image in your dataset has a corresponding label file with the same filename (but with `.txt` extension) in the labels directory.

---

This structure is essential for training YOLO-based object detection models properly.

## Train YOLOv8 Model
Use the Ultralytics API to train the model:
```bash
from ultralytics import YOLO

model = YOLO('yolov8s.pt')  # small variant
model.train(
    data='dataset_red&trans/data3.yaml',
    epochs=110,
    imgsz=640,
    batch=32
)
```

## Orientation Angle Estimation using PCA
Steps used to estimate angle:
-Convert cropped image to grayscale
-Apply thresholding to extract the shape
-Get coordinates of white pixels
-Apply PCA
-Calculate angle from the principal eigenvector
Example code:
```bash
from sklearn.decomposition import PCA
import numpy as np

coords = np.column_stack(np.where(binary_mask > 0))
pca = PCA(n_components=2).fit(coords)
angle = np.arctan2(pca.components_[0, 1], pca.components_[0, 0]) * 180 / np.pi
```


