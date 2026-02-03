"""Run YOLOv8 firearm detector on webcam or video with threat-level overlay."""
from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path

import cv2
from ultralytics import YOLO


COLORS = {
    "low": (0, 255, 255),
    "medium": (0, 165, 255),
    "high": (0, 0, 255),
}


THREAT_THRESHOLDS = {
    "low": 0.5,
    "medium": 0.75,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=str, required=True, help="Path to YOLO weights (.pt)")
    parser.add_argument("--source", type=str, default="0", help="Video path or camera index (default webcam 0)")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for NMS")
    parser.add_argument("--max-det", type=int, default=50, help="Maximum detections per image")
    parser.add_argument("--save", type=str, default="", help="Optional output video path")
    return parser.parse_args()


def open_capture(source: str) -> cv2.VideoCapture:
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
    else:
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video source {source}")
    return cap


def classify_threat(confidence: float) -> tuple[str, tuple[int, int, int]]:
    if confidence < THREAT_THRESHOLDS["low"]:
        return "Low", COLORS["low"]
    if confidence < THREAT_THRESHOLDS["medium"]:
        return "Medium", COLORS["medium"]
    return "High", COLORS["high"]


def main() -> None:
    args = parse_args()
    model = YOLO(args.weights)
    cap = open_capture(args.source)

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.save, fourcc, fps, (width, height))

    fps_queue: deque[float] = deque(maxlen=30)
    try:
        while True:
            grabbed, frame = cap.read()
            if not grabbed:
                break
            start = time.time()
            results = model.predict(
                frame,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                max_det=args.max_det,
                verbose=False,
            )
            end = time.time()
            fps_queue.append(1.0 / max(end - start, 1e-6))
            avg_fps = sum(fps_queue) / len(fps_queue)

            if results:
                boxes = results[0].boxes
                for box in boxes:
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    if cls != 0:
                        continue
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    threat, color = classify_threat(conf)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    label = f"gun {conf:.2f} - {threat}"
                    cv2.putText(frame, label, (int(x1), int(max(0, y1 - 10))), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            if writer:
                writer.write(frame)

            cv2.imshow("Firearm Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
