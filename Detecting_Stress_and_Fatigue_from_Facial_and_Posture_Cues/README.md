#  Detecting Stress and Fatigue from Facial and Posture Cues

## Project Overview
This project focuses on developing a **vision-based system** to detect stress and fatigue using facial cues from video data, with planned extensions to include postural analysis.

The system analyzes subtle, semi-voluntary visual indicators such as eye movement, facial muscle activity, and head motion to infer affective states related to psychosomatic stress and fatigue. This work is grounded in affective computing and healthcare-oriented computer vision.

---

## Motivation & Background
This project is inspired by the work of **Giannakakis et al. (2017)**, which demonstrated that stress and anxiety manifest through measurable facial micro-movements beyond classical facial expressions.

Building upon this insight, the current work establishes a robust **facial emotion detection engine** that can later be extended to incorporate posture-based and temporal stress indicators for real-world deployment.

---

##  Literature Survey

### Stress and Anxiety Detection Using Facial Cues
**Giannakakis et al. (2017)**  
*Stress and anxiety detection using facial cues from videos*  
Biomedical Signal Processing and Control, 31, 89–101  

**Key Insights:**
- Semi-voluntary facial cues (eye activity, mouth movement, head motion) are strong stress indicators  
- Facial expressions alone are insufficient for real-world stress detection  
- Machine learning models can reliably differentiate stress-related states  

These findings motivate the integration of both facial and postural cues in later stages of this project.

---

### Related Research
| Study | Focus | Key Insight |
|-----|-----|-----|
| Li et al. (2020) | Facial fatigue detection | CNNs outperform handcrafted features |
| Kaya et al. (2021) | Face + posture | Multimodal cues improve robustness |
| Zhang et al. (2022) | Temporal modeling | LSTM improves emotion continuity |
| Sharma et al. (2012) | Stress surveys | Multisensor fusion is critical |

---

##  Dataset Used
**AffectNet (YOLO Format)**  
- ~25k images  
- 8 facial emotion classes  
- YOLO-formatted bounding box annotations  

Dataset split:
- Train: 17,101 images  
- Validation: 5,406 images  
- Test: 2,755 images  

---

##  Methodology (Implemented Phase)
1. Dataset verification and cleaning  
2. Facial detection and emotion classification using YOLOv8  
3. Transfer learning with pretrained weights  
4. Automated evaluation using YOLO metrics  

---

##  Training & Results
- Model: YOLOv8n  
- Epochs: 30  
- Image size: 640 × 640  
- GPU: Tesla T4  

**Performance Highlights:**
- Steady reduction in training and validation loss  
- Validation mAP@0.5 ≈ **0.70+**  
- Robust precision and recall across emotion classes  

YOLO automatically generated confusion matrices, F1 curves, PR curves, and per-class metrics, confirming reliable convergence.

---

##  Future Work
- Integrate posture-based cues (shoulder slouch, head tilt)  
- Temporal modeling using LSTM for continuous stress estimation  
- Stress-level regression (low / medium / high)  
- Real-time deployment using Streamlit  

---

##  References
Giannakakis et al. (2017).  
Li et al. (2020).  
Kaya et al. (2021).  
Sharma et al. (2012).  
Zhang et al. (2022).
