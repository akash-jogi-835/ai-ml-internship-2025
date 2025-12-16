# Report: Stress and Fatigue Detection using Facial Cues

## 1. Objective
The objective of this phase of the project was to design and validate a robust **vision-based detection engine** capable of identifying affective states related to stress and fatigue using facial cues. This detection module is intended to serve as a core engine for downstream stress and fatigue analysis systems in healthcare and human monitoring applications.

---

## 2. Dataset Used
**AffectNet (YOLO Format)**  
- Total images: ~25,000  
- Emotion classes: Anger, Contempt, Disgust, Fear, Happy, Neutral, Sad, Surprise  
- Dataset split:
  - Train: 17,101 images  
  - Validation: 5,406 images  
  - Test: 2,755 images  

Each image was annotated in YOLO format with bounding boxes around facial regions and corresponding emotion labels.

---

## 3. Dataset Verification & Preparation
The dataset was programmatically verified in Google Colab:
- Confirmed one-to-one mapping between images and label files  
- Verified correctness of YOLO annotation format  
- Validated directory structure (`train/valid/test → images/labels`)  
- Created and validated a clean `data.yaml` file with absolute paths  

Random samples were visually inspected to ensure bounding boxes and labels aligned correctly with facial regions.

---

## 4. Model & Training Setup
- Model architecture: **YOLOv8n (Nano)**  
- Framework: Ultralytics YOLOv8  
- Training environment: Google Colab (Tesla T4 GPU)  
- Image resolution: 640 × 640  
- Batch size: 16  
- Epochs: 30  
- Optimization: AdamW (auto-selected by YOLO)  
- Transfer learning enabled using pretrained weights  

---

## 5. Training Progress & Evaluation Metrics
During training, the model demonstrated consistent improvement across all key metrics:

- **Training loss** (box, classification, DFL) steadily decreased across epochs  
- **Validation performance improved significantly**, with early epochs showing rapid gains  

By the mid-training stage:
- mAP@0.5 exceeded **0.70**
- mAP@0.5:0.95 reached approximately **0.70**
- Precision and recall improved steadily across classes  

YOLOv8 automatically generated:
- Confusion matrix  
- Precision–Recall curves  
- F1-score curves  
- Training & validation loss plots  
- Per-class evaluation metrics  

All evaluation artifacts were stored in the training output directory for analysis and reporting.

---

## 6. Key Observations
- Facial emotion classes with strong visual cues (Happy, Neutral, Sad) converged faster  
- Subtle expressions (Contempt, Fear) required more epochs for stabilization  
- Transfer learning significantly accelerated convergence  
- The trained model is reliable enough to function as a **feature-extraction and detection backbone** for stress/fatigue pipelines  

---

## 7. Conclusion
This phase successfully delivered a **validated facial emotion detection engine** using transfer learning. The trained YOLOv8 model forms a strong foundation for downstream tasks such as stress-level estimation, fatigue detection, and multimodal fusion with posture or temporal features.

Further extensions (temporal modeling, posture cues, stress regression) can now be built on top of this detection backbone.
