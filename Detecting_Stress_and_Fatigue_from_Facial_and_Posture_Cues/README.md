# 🧠 Detecting Stress and Fatigue from Facial and Posture Cues

### **Project Overview**
This project aims to develop a **computer vision system** capable of detecting **stress** and **fatigue** in individuals using both **facial** and **postural cues** from video input.  
The system will analyze subtle, semi-voluntary visual indicators — such as **eye blinks, mouth activity, and head movement** — along with **body posture deviations** to assess fatigue or psychosomatic stress levels.

This work is inspired by and extends the findings of *Giannakakis et al. (2017)*, who demonstrated that specific facial micro-movements and activity patterns correlate strongly with stress and anxiety states.

---

## 📚 Literature Survey

### **1️⃣ Stress and Anxiety Detection Using Facial Cues**
**Giannakakis, G., Pediaditis, M., Manousos, D., Kazantzaki, E., Chiarugi, F., Simos, P. G., Marias, K., & Tsiknakis, M. (2017).**  
*Stress and anxiety detection using facial cues from videos.* Biomedical Signal Processing and Control, 31, 89–101.  
[https://doi.org/10.1016/j.bspc.2016.06.020](https://doi.org/10.1016/j.bspc.2016.06.020)

🔍 **Key Contributions:**
- Developed a **framework for stress/anxiety detection** using **video-based facial cues**.
- Focused on **semi-voluntary facial features** (eye aperture, blink rate, gaze instability, mouth movement, head motion) rather than explicit facial expressions.
- Proposed an **experimental setup** inducing multiple emotional states (neutral, relaxed, stressed/anxious) through internal and external stressors.
- Applied **machine learning** classifiers to differentiate between stress/anxiety and neutral states with high accuracy.
- Identified **eye activity, mouth activity, head movements**, and **camera-based photoplethysmography (heart rate)** as discriminative features.

📘 **How It Informs Our Project:**
This paper establishes that stress-related visual signals exist beyond classical facial expressions — a key insight driving our inclusion of **postural features** (slouching, head tilt, shoulder droop).  
Our model aims to integrate both **facial micro-cues** and **macro postural cues** to enhance real-world robustness in unconstrained environments.

---

### **2️⃣ Related Research References**

| Study | Focus Area | Key Insight |
|-------|-------------|-------------|
| **Li, X., Sun, X., et al. (2020).** *Detecting mental fatigue using facial features and deep learning.* IEEE Access. | Used CNNs to analyze eye and mouth dynamics for fatigue classification. | Deep learning outperforms handcrafted features for emotion-based fatigue detection. |
| **Kaya, H., et al. (2021).** *A multimodal approach for stress detection using posture and facial expressions.* Pattern Recognition Letters. | Combined facial and skeletal features for stress classification. | Demonstrates complementarity of face + posture data. |
| **Zhang, Y. et al. (2022).** *Real-time emotion recognition from facial landmarks using CNN and LSTM networks.* Neurocomputing. | Used facial landmarks and sequential models. | Supports temporal modeling for continuous emotion recognition. |
| **Sharma, N. et al. (2012).** *Objective measures, sensors and computational techniques for stress recognition and classification: a survey.* *Comput. Methods Programs Biomed.* | Reviewed physiological and behavioral stress detection methods. | Highlights multi-sensor fusion (EEG, GSR, facial cues). |

---

## 📊 Dataset Design Plan

Inspired by **Giannakakis et al. (2017)**, our dataset will incorporate both **facial** and **postural cues** under controlled and natural conditions.

### **1️⃣ Data Sources**
- **Public Datasets (Facial Cues):**
  - [AffectNet](http://mohammadmahoor.com/affectnet/) – Facial emotion dataset with diverse expressions.
  - [DFEW](https://dfew-dataset.github.io/) – Dynamic Facial Expressions in the Wild.
  - [FER2013](https://www.kaggle.com/datasets/deadskull7/fer2013) – Basic emotion dataset (can fine-tune for fatigue-like states).

- **Custom Dataset (Posture + Fatigue Simulation):**
  - Record videos of participants simulating **relaxed**, **alert**, and **fatigued/stressed** states.
  - Capture **upper-body frames** focusing on shoulders, head, and posture tilt.
  - Annotate with class labels: `"Neutral"`, `"Stressed"`, `"Fatigued"`.

### **2️⃣ Experimental Design**
| Condition | Stimulus | Expected Behavior | Captured Features |
|------------|-----------|-------------------|-------------------|
| Neutral | Calm environment | Normal posture, minimal blinking | Baseline |
| Stressed | Task under time pressure | Faster blinking, furrowed brow, tensed shoulders | Eye aperture, head movement |
| Fatigued | Prolonged sitting/late hours | Slouching, slow eye movement, yawning | Posture angle, mouth activity |

### **3️⃣ Feature Focus**
- **Facial Cues:** Eye aperture, blink rate, mouth openness, head rotation.
- **Postural Cues:** Shoulder slouch angle, head tilt, back curvature.
- **Derived Metrics:** Eye aspect ratio (EAR), mouth aspect ratio (MAR), posture deviation (Pdev).

---

## 🧠 Methodology (Proposed)

1. **Face & Pose Detection** – YOLOv8 or MediaPipe.  
2. **Feature Extraction** – EAR, MAR, pose keypoints.  
3. **Model Training (Transfer Learning)** – Fine-tune ResNet50 / EfficientNet for classification.  
4. **Temporal Modeling (Optional)** – Add LSTM layer for time-series analysis of stress progression.  
5. **Evaluation** – Accuracy, Precision, F1-score, and confusion matrix.  

---

## 📘 Expected Outcomes
- Dataset with annotated facial and posture stress cues.
- Model capable of detecting **fatigue and stress levels in real time**.
- Improved robustness through combined micro (face) and macro (posture) feature analysis.
- Strong theoretical grounding in prior literature (esp. Giannakakis et al., 2017).

---

## 🔖 References

1. Giannakakis, G., Pediaditis, M., Manousos, D., Kazantzaki, E., Chiarugi, F., Simos, P. G., Marias, K., & Tsiknakis, M. (2017). *Stress and anxiety detection using facial cues from videos.* Biomedical Signal Processing and Control, 31, 89–101. [https://doi.org/10.1016/j.bspc.2016.06.020](https://doi.org/10.1016/j.bspc.2016.06.020)

2. Li, X., Sun, X., et al. (2020). *Detecting mental fatigue using facial features and deep learning.* IEEE Access.

3. Kaya, H., et al. (2021). *A multimodal approach for stress detection using posture and facial expressions.* Pattern Recognition Letters.

4. Sharma, N., Gedeon, T., & Aldrich, C. (2012). *Objective measures, sensors and computational techniques for stress recognition and classification: a survey.* *Computer Methods and Programs in Biomedicine, 108*(3), 1287–1301.

5. Zhang, Y. et al. (2022). *Real-time emotion recognition from facial landmarks using CNN and LSTM networks.* Neurocomputing.
