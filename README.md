# Real-Time Firearm Detection Pipeline

End-to-end YOLOv8 workflow for training a firearm detector by combining the Monash Guns Dataset (MGD) and the US Real-time Gun Detection in CCTV (USRT) dataset. The focus is producing a CCTV-ready model with real-time inference overlays and evaluation utilities for a course project.

## Project layout
```
├── dataset/                     # Final merged YOLO dataset
├── dataset_mgd/                 # Intermediate MGD splits (train/test)
├── dataset_usrt/                # Intermediate USRT samples prior to splitting
├── runs_firearm/                # Training runs and artifacts
├── scripts/
│   ├── convert_mgd_to_yolo.py   # MGD VOC → YOLO conversion
│   ├── convert_usrt_to_yolo.py  # USRT annotations → YOLO conversion
│   ├── split_dataset.py         # Train/val split
│   ├── train_yolov8.py          # Ultralytics training helper
│   ├── evaluate_results.py      # Validation metrics
│   ├── infer_video.py           # Real-time/video inference with threat levels
│   └── visualize_samples.py     # Visualize plots
├── firearm.yaml                 # YOLO dataset config (two classes: handgun and pistol are combined, short_rifle)
├── requirements.txt
└── README.md
```

## Setup
1. (Optional) create a virtual environment to use.

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Download and extract the datasets:
   - **MGD (Monash Guns Dataset)** – https://github.com/MarcusLimJunYi/Monash-Guns-Dataset/
   - **USRT (US Real-time Gun Detection in CCTV)** – https://github.com/Deepknowledge-US/US-Real-time-gun-detection-in-CCTV-An-open-problem-dataset

## Data preparation workflow
Run the following scripts from the project root in order:

1. **Convert MGD to YOLO format**
   ```
   python scripts/convert_mgd_to_yolo.py
   ```
2. **Convert USRT to YOLO format**
   ```
   python scripts/convert_usrt_to_yolo.py
   ```
3. **Merge splits**
   ```
   python scripts/split_dataset.py --val-ratio 0.2 --seed 42
   ```
   This command copies the MGD train/test subsets (stored under `dataset_mgd/`) into the final `dataset/` directory and splits/merges the USRT samples (`dataset_usrt/`) then copies into the final `dataset/` directory. Adjust `--val-ratio` if you want a different USRT split.
   
4. (Optional) Visualize a random subset to check ground truth:
   ```
   python scripts/visualize_samples.py --subset train --count 8 --save-dir samples
   ```

Each script prints a summary of copied images for verification
```
dataset/
├── images/{train,val}/
└── labels/{train,val}/
```

## Training
Train a YOLOv8 model:
```
python scripts/train_yolov8.py --epochs 100 --batch 64 --imgsz 640 --device 0 #if cuda is available
```
All runs are stored under `runs_firearm/detect/yolov8s_mgd_usrt/`. Adjust hyperparameters through CLI flags.

## Evaluation
Evaluate the latest checkpoint on the validation split (runs automatically after training):
```
python scripts/evaluate_results.py --weights runs_firearm/detect/yolov8s_mgd_usrt/weights/best.pt
```
Printed metrics include `mAP@0.5`, `mAP@0.5:0.95`, precision, and recall in order to compare different training runs

## Real-time / video inference
Use a webcam (`--source 0`) or any video file to test the trained detector with threat-level overlays:
```
python scripts/infer_video.py \
  --weights runs_firearm/detect/yolov8s_mgd_usrt/weights/best.pt \
  --source 0 \
  --conf 0.35
```
Threat color scheme used in the overlay (default `--conf 0.25`):
- **Yellow**: Low threat (<0.5 confidence)
- **Orange**: Medium threat (0.5–0.75)
- **Red**: High threat (>0.75)

Add `--save out.mp4` to export annotated footage.

### Creating a test clip from the USRT frames
USRT ships as individual frames that can be stitched together for live inference:

```
# 1) Create the clip
mkdir -p data
ffmpeg -framerate 2 -pattern_type glob \
  -i "/home/dev2/Desktop/CAP 5415 Project/data/USRT 2FPS/Images/Cam1-From09-23-00To10-03-25_Segment_0_x264_frame_*.jpg" \
  -c:v libx264 -pix_fmt yuv420p data/usrt_clip.mp4

# 2) Run the detector on that clip
python scripts/infer_video.py \
  --weights runs_firearm/yolov8s_mgd_usrt/weights/best.pt \
  --source data/usrt_clip.mp4 \
  --conf 0.35 --iou 0.5 --max-det 50 \
  --save annotated_usrt_clip.mp4
```
## Notes & limitations
- Inference speed, and performance depends heavily on GPU resources and how closely test footage matches the training domain.
- Datasets contain mock scenarios and synthetic footage as such it may contain gaps in capabilities
- This project is for academic research only.
