# ================================
# YOLOv8 SEGMENTATION SCRIPT
# Clean Version for IIITH-INTERNSHIP Project
# ================================

from ultralytics import YOLO
import os
import cv2
import json
import pandas as pd
from pathlib import Path
import argparse
from datetime import datetime


def setup_directories():
    """Create necessary directories for segmentation outputs"""
    base_dir = Path("runs/segment")
    (base_dir / "images").mkdir(parents=True, exist_ok=True)
    (base_dir / "masks").mkdir(parents=True, exist_ok=True)
    (base_dir / "reports").mkdir(parents=True, exist_ok=True)
    return base_dir


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="YOLOv8 Image Segmentation")
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n-seg.pt",
        help="YOLO segmentation model path (default: yolov8n-seg.pt)"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="images",
        help="Source directory for input images (default: images)"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25)"
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.7,
        help="IOU threshold for NMS (default: 0.7)"
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="Inference image size (default: 640)"
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Save results to JSON file"
    )
    return parser.parse_args()


class YOLOSegmenter:
    def __init__(self, model_path, confidence=0.25, iou=0.7, img_size=640):
        """Initialize YOLO segmenter"""
        try:
            self.model = YOLO(model_path)
            self.confidence = confidence
            self.iou = iou
            self.img_size = img_size
            self.segmentation_stats = {
                "total_images": 0,
                "processed_images": 0,
                "failed_images": 0,
                "total_segments": 0,
                "class_distribution": {}
            }
            print(f"✅ Model loaded successfully: {model_path}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise

    def process_image(self, image_path):
        """Run segmentation on a single image"""
        try:
            results = self.model.predict(
                source=image_path,
                conf=self.confidence,
                iou=self.iou,
                imgsz=self.img_size,
                save=False,
                save_txt=False
            )
            return results[0] if results else None
        except Exception as e:
            print(f"⚠️ Error processing {image_path}: {e}")
            return None

    def save_segmented_image(self, result, image_path, output_dir):
        """Save image with segmentation masks"""
        try:
            plotted_image = result.plot()
            output_path = os.path.join(output_dir, f"segmented_{Path(image_path).name}")
            cv2.imwrite(output_path, plotted_image)
            return output_path
        except Exception as e:
            print(f"⚠️ Error saving segmented image: {e}")
            return None

    def save_masks(self, result, image_path, masks_dir):
        """Save individual masks"""
        if not hasattr(result, "masks") or result.masks is None:
            print(f"⚠️ No masks found for {Path(image_path).name}")
            return

        os.makedirs(masks_dir, exist_ok=True)
        for idx, mask in enumerate(result.masks.data):
            mask_img = (mask.cpu().numpy() * 255).astype("uint8")
            mask_path = os.path.join(masks_dir, f"{Path(image_path).stem}_mask_{idx}.png")
            cv2.imwrite(mask_path, mask_img)
            self.segmentation_stats["total_segments"] += 1

    def update_class_stats(self, result):
        """Update segmentation class distribution"""
        if hasattr(result, "boxes") and result.boxes is not None:
            for box in result.boxes:
                cls = int(box.cls[0])
                class_name = self.model.names[cls]
                self.segmentation_stats["class_distribution"][class_name] = (
                    self.segmentation_stats["class_distribution"].get(class_name, 0) + 1
                )

    def generate_report(self, output_dir):
        """Generate a summary report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "segmentation_statistics": self.segmentation_stats,
            "settings": {
                "confidence_threshold": self.confidence,
                "iou_threshold": self.iou,
                "image_size": self.img_size
            }
        }

        report_path = os.path.join(output_dir, "reports", "segmentation_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        print("\n" + "=" * 50)
        print("📊 SEGMENTATION SUMMARY")
        print("=" * 50)
        print(f"Total images processed: {self.segmentation_stats['processed_images']}")
        print(f"Total segments created: {self.segmentation_stats['total_segments']}")
        print(f"Failed images: {self.segmentation_stats['failed_images']}")
        print("\nClass distribution:")
        for class_name, count in self.segmentation_stats["class_distribution"].items():
            print(f"  {class_name}: {count}")
        print(f"\n📝 Report saved to: {report_path}")


def main():
    args = parse_arguments()
    output_dir = setup_directories()

    segmenter = YOLOSegmenter(
        model_path=args.model,
        confidence=args.conf,
        iou=args.iou,
        img_size=args.img_size
    )

    # 🔧 Collect input images
    if os.path.isfile(args.source):
        image_paths = [args.source]
    elif os.path.isdir(args.source):
        image_paths = [
            os.path.join(args.source, img_name)
            for img_name in os.listdir(args.source)
            if img_name.lower().endswith((".jpg", ".png", ".jpeg", ".bmp", ".tiff"))
        ]
    else:
        print(f"❌ Source path does not exist: {args.source}")
        return

    if not image_paths:
        print("⚠️ No images found to process.")
        return

    segmenter.segmentation_stats["total_images"] = len(image_paths)
    print(f"🚀 Starting segmentation on {len(image_paths)} images...\n")

    for image_path in image_paths:
        print(f"🎨 Processing: {Path(image_path).name}")
        result = segmenter.process_image(image_path)

        if result is not None:
            image_output = segmenter.save_segmented_image(result, image_path, "runs/segment/images")
            masks_output_dir = "runs/segment/masks"
            segmenter.save_masks(result, image_path, masks_output_dir)
            segmenter.update_class_stats(result)
            print(f"✅ Segmentation completed: {Path(image_output).name}")
            segmenter.segmentation_stats["processed_images"] += 1
        else:
            print(f"❌ Failed to process: {Path(image_path).name}")
            segmenter.segmentation_stats["failed_images"] += 1

    # After segmentation loop, save summary to CSV
    csv_path = Path("runs/segment/train/results.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    summary_data = {
        "timestamp": [datetime.now().isoformat()],
        "total_images": [segmenter.segmentation_stats.get("total_images", 0)],
        "processed_images": [segmenter.segmentation_stats.get("processed_images", 0)],
        "failed_images": [segmenter.segmentation_stats.get("failed_images", 0)],
        "total_segments": [segmenter.segmentation_stats.get("total_segments", 0)],
    }

    # Add class distribution dynamically
    for class_name, count in segmenter.segmentation_stats.get("class_distribution", {}).items():
        summary_data[class_name] = [count]

    pd.DataFrame(summary_data).to_csv(csv_path, index=False)
    print(f"\nSegmentation results saved to: {csv_path}")

    segmenter.generate_report("runs/segment")
    print("\nSegmentation completed! Results saved in 'runs/segment/' directory.")


if __name__ == "__main__":
    main()
