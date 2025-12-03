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
    )
    best_path = Path(results.save_dir) / "weights" / "best.pt"
    print(f"Training complete. Best model saved to {best_path}")


if __name__ == "__main__":
    main()
