from ultralytics import YOLO
import os
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Configuration
# -----------------------------
MODEL_PATH = "yolov8n-seg.pt"   # segmentation model
INPUT_FOLDER = "input_frames"
OUTPUT_FOLDER = "processed_frames"
DETECTED_FOLDER = "output_frames/run1"  # YOLO will store annotated frames here

# -----------------------------
# Setup
# -----------------------------
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️ Using device: {device}")

# Load YOLO model
model = YOLO(MODEL_PATH)

# -----------------------------
# Step 1: Process each frame and collect metrics
# -----------------------------
valid_exts = (".jpg", ".jpeg", ".png", ".bmp")
image_files = sorted([f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(valid_exts)])

if not image_files:
    print("🚫 No images found in 'input_frames' folder.")
    exit()

print(f"\n🟢 Found {len(image_files)} frames. Starting segmentation and metric collection...")

# Lists to collect frame-wise metrics
frame_indices = []
avg_confidences = []
detections_per_frame = []
seg_area_ratios = []

for idx, img_name in enumerate(tqdm(image_files, desc="Processing frames")):
    input_path = os.path.join(INPUT_FOLDER, img_name)
    output_path = os.path.join(OUTPUT_FOLDER, img_name)

    # Run prediction + save annotated frame
    results = model.predict(
        source=input_path,
        device=device,
        save=True,                  # ✅ Save annotated frame
        project="output_frames",     # ✅ Root output folder
        name="run1",                 # ✅ Subfolder for this run
        exist_ok=True,
        verbose=False,
        conf=0.5
    )

    # Save a copy of processed frame (optional)
    results[0].save(filename=output_path)

    # Extract per-frame metrics
    boxes = results[0].boxes
    masks = results[0].masks

    confs = [float(b.conf) for b in boxes] if boxes is not None else []
    avg_conf = np.mean(confs) if confs else 0
    num_detections = len(confs)

    # Compute approximate segmentation area ratio
    if masks is not None and masks.data is not None:
        mask_area = float(masks.data.sum())  # total pixels in masks
        total_area = masks.data.numel()      # total pixels in frame
        seg_ratio = mask_area / total_area
    else:
        seg_ratio = 0

    # Append metrics
    frame_indices.append(idx + 1)
    avg_confidences.append(avg_conf)
    detections_per_frame.append(num_detections)
    seg_area_ratios.append(seg_ratio)

print(f"\n✅ Processing complete! Processed images saved in '{DETECTED_FOLDER}' folder.")

# -----------------------------
# Step 2: Plot frame-wise metrics
# -----------------------------
plt.figure(figsize=(10, 6))

plt.subplot(3, 1, 1)
plt.plot(frame_indices, avg_confidences, marker='o', color='blue')
plt.title("Average Confidence per Frame")
plt.ylabel("Avg Confidence")
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(frame_indices, detections_per_frame, marker='o', color='green')
plt.title("Number of Detections per Frame")
plt.ylabel("Detections")
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(frame_indices, seg_area_ratios, marker='o', color='orange')
plt.title("Segmentation Area Ratio per Frame")
plt.xlabel("Frame Index")
plt.ylabel("Area Ratio")
plt.grid(True)

plt.tight_layout()
plt.savefig("frame_metrics_graph.png")
plt.close()

print("📊 Frame metrics plotted and saved as 'frame_metrics_graph.png'")

# -----------------------------
# Step 3: Summary statistics
# -----------------------------
print("\n📈 Summary of Frame Metrics:")
print(f"Average confidence across all frames: {np.mean(avg_confidences):.4f}")
print(f"Average detections per frame: {np.mean(detections_per_frame):.2f}")
print(f"Average segmentation area ratio: {np.mean(seg_area_ratios):.4f}")

# -----------------------------
# Step 4: Compute heuristic metrics
# -----------------------------
total_frames = len(image_files)
frames_with_detections = sum(1 for c in avg_confidences if c > 0)
total_detections = sum(detections_per_frame)
confident_detections = sum(1 for c in avg_confidences if c > 0.7)

pseudo_precision = confident_detections / total_detections if total_detections else 0
pseudo_recall = frames_with_detections / total_frames
pseudo_accuracy = (pseudo_precision + pseudo_recall) / 2

print("\n🧠 Heuristic Evaluation Metrics (no ground truth):")
print(f" Precision (pseudo): {pseudo_precision:.3f}")
print(f" Recall (pseudo):    {pseudo_recall:.3f}")
print(f" Accuracy (pseudo):  {pseudo_accuracy:.3f}")

# -----------------------------
# Step 5: Plot heuristic metrics as bar graph
# -----------------------------
metrics = ['Precision', 'Recall', 'Accuracy']
values = [pseudo_precision, pseudo_recall, pseudo_accuracy]

plt.figure(figsize=(6, 4))
bars = plt.bar(metrics, values, color=['#1f77b4', '#2ca02c', '#ff7f0e'])
plt.title('Heuristic Evaluation Metrics (No Ground Truth)')
plt.ylim(0, 1)
plt.ylabel('Score')

# Add value labels on bars
for bar, val in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width()/2, val + 0.02, f"{val:.2f}", ha='center', fontsize=10)

plt.tight_layout()
plt.savefig("heuristic_metrics_graph.png")
plt.show()

print("📈 Heuristic metrics plotted and saved as 'heuristic_metrics_graph.png'")
