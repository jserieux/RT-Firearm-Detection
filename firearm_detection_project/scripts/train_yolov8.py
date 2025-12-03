"""Train YOLOv8 on the merged firearm dataset."""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=str, default="firearm.yaml", help="Dataset config file")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--project", type=str, default="runs_firearm")
    parser.add_argument("--name", type=str, default="yolov8s_mgd_usrt")
    parser.add_argument("--weights", type=str, default="yolov8s.pt")
    parser.add_argument("--optimizer", type=str, default="auto")
    parser.add_argument("--hsv-h", type=float, default=0.015)
    parser.add_argument("--hsv-s", type=float, default=0.7)
    parser.add_argument("--hsv-v", type=float, default=0.4)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--translate", type=float, default=0.1)
    parser.add_argument("--mosaic", type=float, default=1.0)
    parser.add_argument("--copy-paste", type=float, default=0.0)
    parser.add_argument("--erasing", type=float, default=0.4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.weights)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        optimizer=args.optimizer,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        scale=args.scale,
        translate=args.translate,
        mosaic=args.mosaic,
        copy_paste=args.copy_paste,
        erasing=args.erasing,
    )
    best_path = Path(results.save_dir) / "weights" / "best.pt"
    print(f"Training complete. Best model saved to {best_path}")


if __name__ == "__main__":
    main()
