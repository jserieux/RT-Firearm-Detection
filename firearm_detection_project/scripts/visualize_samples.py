"""Visualize random labeled samples to verify YOLO annotations."""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


def load_labels(label_path: Path) -> list[tuple[float, float, float, float]]:
    boxes: list[tuple[float, float, float, float]] = []
    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            _, x_c, y_c, w, h = parts
            boxes.append((float(x_c), float(y_c), float(w), float(h)))
    return boxes


def visualize_samples(dataset_dir: Path, subset: str, count: int, save_dir: Path | None) -> None:
    images_dir = dataset_dir / "images" / subset
    labels_dir = dataset_dir / "labels" / subset
    images = [p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    if not images:
        raise FileNotFoundError(f"No images found in {images_dir}")
    random.shuffle(images)
    selected = images[: min(count, len(images))]

    cols = min(4, len(selected))
    rows = (len(selected) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax, img_path in zip(axes, selected):
        label_path = labels_dir / f"{img_path.stem}.txt"
        ax.axis("off")
        if not label_path.exists():
            ax.set_title("No label")
            ax.imshow(Image.open(img_path))
            continue
        img = Image.open(img_path).convert("RGB")
        ax.imshow(img)
        width, height = img.size
        for x_c, y_c, w, h in load_labels(label_path):
            x = (x_c - w / 2) * width
            y = (y_c - h / 2) * height
            rect = patches.Rectangle((x, y), w * width, h * height, linewidth=2, edgecolor="red", facecolor="none")
            ax.add_patch(rect)
        ax.set_title(img_path.name)

    for idx in range(len(selected), len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        out_path = save_dir / "samples.png"
        plt.savefig(out_path)
        print(f"Saved visualization to {out_path}")
    else:
        plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--subset", type=str, default="train", choices=["train", "val"])
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--save-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    visualize_samples(args.dataset, args.subset, args.count, args.save_dir)


if __name__ == "__main__":
    main()
