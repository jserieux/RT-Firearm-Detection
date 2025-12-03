"""Split prepared dataset into train/val subsets."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import random
import shutil

from tqdm import tqdm

LOGGER = logging.getLogger("split_dataset")


def split_dataset(project_root: Path, val_ratio: float, seed: int) -> None:
    rng = random.Random(seed)
    images_train_dir = project_root / "dataset/images/train"
    labels_train_dir = project_root / "dataset/labels/train"
    images_val_dir = project_root / "dataset/images/val"
    labels_val_dir = project_root / "dataset/labels/val"
    for path in (images_val_dir, labels_val_dir):
        path.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(images_train_dir.glob("*"))
    paired = [p for p in image_paths if (labels_train_dir / (p.stem + ".txt")).exists()]
    if not paired:
        LOGGER.warning("No paired image/label files found; did you run conversion scripts?")
        return

    rng.shuffle(paired)
    val_count = max(1, int(len(paired) * val_ratio))
    val_images = set(paired[:val_count])

    LOGGER.info("Splitting %s images into %s train / %s val", len(paired), len(paired) - val_count, val_count)

    for img_path in tqdm(paired, desc="Moving files"):
        label_path = labels_train_dir / (img_path.stem + ".txt")
        if img_path in val_images:
            dst_img = images_val_dir / img_path.name
            dst_lbl = labels_val_dir / label_path.name
        else:
            dst_img = images_train_dir / img_path.name
            dst_lbl = labels_train_dir / label_path.name
        if img_path != dst_img:
            shutil.move(str(img_path), dst_img)
        if label_path != dst_lbl:
            shutil.move(str(label_path), dst_lbl)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    split_dataset(args.root, args.val_ratio, args.seed)


if __name__ == "__main__":
    main()
