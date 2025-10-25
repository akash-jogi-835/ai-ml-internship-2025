from ultralytics import YOLO
import os
import cv2
import json
import pandas as pd
from pathlib import Path
import argparse
from datetime import datetime

def setup_directories():
    """Create necessary directories for YOLO detection results"""
    base_dir = Path(__file__).resolve().parent
    detect_dir = base_dir / "runs" / "detect"
    (detect_dir / "images").mkdir(parents=True, exist_ok=True)
    (detect_dir / "labels").mkdir(parents=True, exist_ok=True)
    (detect_dir / "reports").mkdir(parents=True, exist_ok=True)
    return detect_dir

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='YOLOv8 Object Detection')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                       help='YOLO model path (default: yolov8n.pt)')
    parser.add_argument('--source', type=str, default='images',
                       help='Source directory for input images (default: images/)')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold (default: 0.25)')
    parser.add_argument('--iou', type=float, default=0.7,
                       help='IOU threshold for NMS (default: 0.7)')
    parser.add_argument('--img-size', type=int, default=640,
                       help='Inference size (default: 640)')
    parser.add_argument('--save-txt', action='store_true',
                       help='Save results to text file')
    parser.add_argument('--save-json', action='store_true',
                       help='Save results to JSON file')
    return parser.parse_args()

class YOLODetector:
    def __init__(self, model_path, confidence=0.25, iou=0.7, img_size=640):
        """Initialize YOLO detector"""
        try:
            self.model = YOLO(model_path)
            self.confidence = confidence
            self.iou = iou
            self.img_size = img_size
            self.detection_stats = {
                'total_images': 0,
                'processed_images': 0,
                'failed_images': 0,
                'total_detections': 0,
                'class_distribution': {}
            }
            print(f"Model loaded successfully: {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def process_image(self, image_path):
        """Process a single image and return results"""
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
            print(f"Error processing {image_path}: {e}")
            return None

    def save_detection_image(self, result, image_path, output_dir):
        """Save image with detection boxes"""
        try:
            plotted_image = result.plot()
            output_path = output_dir / "images" / f"detected_{Path(image_path).name}"
            cv2.imwrite(str(output_path), plotted_image)
            return output_path
        except Exception as e:
            print(f"Error saving detection image: {e}")
            return None

    def save_detection_data(self, result, image_path, output_dir):
        """Save detection data to text and JSON files"""
        image_name = Path(image_path).stem
        labels_dir = output_dir / "labels"
        labels_dir.mkdir(exist_ok=True)

        if hasattr(result, 'boxes') and result.boxes is not None:
            txt_path = labels_dir / f"{image_name}.txt"
            with open(txt_path, 'w') as f:
                for box in result.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    coords = box.xywhn[0].tolist()
                    f.write(f"{cls} {coords[0]} {coords[1]} {coords[2]} {coords[3]} {conf}\n")
                    self.detection_stats['total_detections'] += 1
                    class_name = self.model.names[cls]
                    self.detection_stats['class_distribution'][class_name] = \
                        self.detection_stats['class_distribution'].get(class_name, 0) + 1

    def generate_report(self, output_dir):
        """Generate a summary report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'detection_statistics': self.detection_stats,
            'settings': {
                'confidence_threshold': self.confidence,
                'iou_threshold': self.iou,
                'image_size': self.img_size
            }
        }
        report_path = output_dir / "reports" / "detection_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print("\n" + "="*50)
        print("DETECTION SUMMARY")
        print("="*50)
        print(f"Total images processed: {self.detection_stats['processed_images']}")
        print(f"Total detections: {self.detection_stats['total_detections']}")
        print(f"Failed images: {self.detection_stats['failed_images']}")
        print("\nClass distribution:")
        for class_name, count in self.detection_stats['class_distribution'].items():
            print(f"  {class_name}: {count}")
        print(f"\nReport saved to: {report_path}")

def main():
    args = parse_arguments()
    base_dir = Path(__file__).resolve().parent
    output_dir = setup_directories()

    # Ensure model and image paths are absolute
    args.model = str((base_dir / args.model).resolve())
    args.source = str((base_dir / args.source).resolve())

    detector = YOLODetector(
        model_path=args.model,
        confidence=args.conf,
        iou=args.iou,
        img_size=args.img_size
    )

    # Collect image paths
    if os.path.isfile(args.source):
        image_paths = [args.source]
    elif os.path.isdir(args.source):
        image_paths = [
            os.path.join(args.source, img_name)
            for img_name in os.listdir(args.source)
            if img_name.lower().endswith((".jpg", ".png", ".jpeg", ".bmp", ".tiff"))
        ]
    else:
        print(f"Source path does not exist: {args.source}")
        return

    if not image_paths:
        print("No images found to process.")
        return

    detector.detection_stats['total_images'] = len(image_paths)
    print(f"Starting detection on {len(image_paths)} images...\n")

    for image_path in image_paths:
        print(f"Processing: {Path(image_path).name}")
        result = detector.process_image(image_path)

        if result is not None:
            output_path = detector.save_detection_image(result, image_path, output_dir)
            if output_path:
                print(f"Detection saved: {output_path.name}")
            if args.save_txt or args.save_json:
                detector.save_detection_data(result, image_path, output_dir)
            detector.detection_stats['processed_images'] += 1
        else:
            print(f"Failed to process: {Path(image_path).name}")
            detector.detection_stats['failed_images'] += 1

    # After detection loop ends, save detection summary to CSV
    csv_path = output_dir / "train" / "results.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    summary_data = {
        "timestamp": [datetime.now().isoformat()],
        "total_images": [detector.detection_stats.get("total_images", 0)],
        "processed_images": [detector.detection_stats.get("processed_images", 0)],
        "failed_images": [detector.detection_stats.get("failed_images", 0)],
        "total_detections": [detector.detection_stats.get("total_detections", 0)],
    }

    # Add class distribution dynamically
    for class_name, count in detector.detection_stats.get("class_distribution", {}).items():
        summary_data[class_name] = [count]

    pd.DataFrame(summary_data).to_csv(csv_path, index=False)
    print(f"\nResults CSV generated at: {csv_path}")

    detector.generate_report(output_dir)
    print("\nDetection completed! Results saved in 'runs/detect/' directory.")

if __name__ == "__main__":
    main()
