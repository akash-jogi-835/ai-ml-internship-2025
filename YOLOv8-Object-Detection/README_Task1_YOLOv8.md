# 🧠 Task-1: YOLOv8 Detection and Segmentation Analysis

## 📌 Overview
This task focuses on training, analyzing, and visualizing **YOLOv8-based Detection and Segmentation models** using organized folder structures and enhanced Python utilities.  
The objective was to:
- Perform object detection and segmentation on given datasets.  
- Collect and store model metrics in structured directories.  
- Analyze results through precision, recall, F1, and per-class distributions.  
- Visualize performance metrics for detailed evaluation.  

---

## 📂 Project Structure
```
IIITH-INTERNSHIP/
│
├── images/                          # Input images for inference or evaluation
│
├── runs/
│   ├── detect/
│   │   ├── images/                  # Detection output images
│   │   ├── labels/                  # Detection label files
│   │   ├── reports/
│   │   │   └── detection_report.json
│   │   └── train/
│   │       └── results.csv          # Detection metrics summary
│   │
│   └── segment/
│       ├── images/                  # Segmentation output images
│       ├── masks/                   # Generated segmentation masks
│       ├── reports/
│       └── train/
│           └── results.csv          # Segmentation metrics summary
│
├── detect_multiple.py               # Runs YOLOv8 detection on multiple images
├── segment_multiple.py              # Runs YOLOv8 segmentation on multiple images
├── analyze_metrics.py               # Analyzes and visualizes YOLO metrics
├── yolov8n.pt                       # Pre-trained YOLOv8 detection model
├── yolov8n-seg.pt                   # Pre-trained YOLOv8 segmentation model
└── yoloenv/                         # Conda environment for YOLO operations
```

---

## ⚙️ Step-by-Step Workflow

### 🧩 Step 1: Environment Setup
Create and activate the YOLOv8 environment:
```bash
conda create -n yoloenv python=3.10
conda activate yoloenv
pip install ultralytics matplotlib pandas
```

### 🧠 Step 2: Detection on Multiple Images
Run the detection script to process all images:
```bash
python detect_multiple.py
```
- Automatically processes images in `/images`  
- Saves results to `/runs/detect/`  

### 🎨 Step 3: Segmentation on Multiple Images
Run the segmentation pipeline:
```bash
python segment_multiple.py
```
- Generates masks in `/runs/segment/masks/`  
- Saves results and logs in `/runs/segment/train/results.csv`  

### 📈 Step 4: Metrics Analysis
Use the analyzer to review performance:
```bash
python analyze_metrics.py
```
- Reads metrics from:
  - `runs/detect/train/results.csv`
  - `runs/segment/train/results.csv`
- Automatically detects if the CSV contains YOLO logs or custom summaries
- Produces:
  - **Per-class distribution bar charts**
  - **Epoch-based performance graphs** (Precision, Recall, F1, mAP)

---

## 📊 Example Visualization

### Precision, Recall, and F1 per Class (YOLOv8)
![PRF1 Chart](example_metrics.png)

### Class Distribution (Custom Detection Summary)
Bar charts are generated dynamically for processed images and detected objects.

---

## 🧾 Output Summary Example

```
SUMMARY OF DETECTION TRAINING RESULTS
======================================================================
Timestamp: 2025-10-25T18:47:36
Total Images: 32
Processed: 31
Failed: 1

SUMMARY OF SEGMENTATION TRAINING RESULTS
======================================================================
Timestamp: 2025-10-25T19:02:10
Total Images: 20
Processed: 20
Failed: 0
```

---

## 🧩 Key Features
- 📦 Organized folder structure for detection and segmentation runs  
- 📊 Auto-detection of CSV formats (YOLO or custom summaries)  
- 🧠 Robust metric visualization (Precision, Recall, F1, mAP)  
- 🧾 Auto-generated textual summaries and JSON reports  
- 🔍 Modular design for future integration with MLOps workflows  

---

## 🚀 Next Steps
- Integrate TensorBoard or Weights & Biases for advanced metric tracking.  
- Automate report generation in `.txt` and `.pdf` formats.  
- Extend to multi-class segmentation with annotation heatmaps.  

---

## 🧑‍💻 Author
**Gowri Akash**  
IIITH Internship – Task-1  
*YOLOv8 Detection and Segmentation Analysis Pipeline*
