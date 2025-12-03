"""Assemble final train/val splits by merging MGD and USRT samples."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import random
import shutil

from tqdm import tqdm

LOGGER = logging.getLogger("split_dataset")


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_split(src_images: Path, src_labels: Path, dst_images: Path, dst_labels: Path) -> int:
    count = 0
    for img_path in sorted(src_images.glob("*")):
        if not img_path.is_file():
            continue
        label_path = src_labels / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue
        shutil.copy2(img_path, dst_images / img_path.name)
        shutil.copy2(label_path, dst_labels / label_path.name)
        count += 1
    return count


def split_usrt(usrt_root: Path, val_ratio: float, seed: int) -> None:
    rng = random.Random(seed)
    train_img = usrt_root / "images/train"
    train_lbl = usrt_root / "labels/train"
    val_img = usrt_root / "images/val"
    val_lbl = usrt_root / "labels/val"
    val_img.mkdir(parents=True, exist_ok=True)
    val_lbl.mkdir(parents=True, exist_ok=True)

    paired = [p for p in sorted(train_img.glob("*")) if (train_lbl / f"{p.stem}.txt").exists()]
    if not paired:
        LOGGER.warning("No USRT samples found to split. Did you run convert_usrt_to_yolo.py?")
        return

    rng.shuffle(paired)
    val_count = max(1, int(len(paired) * val_ratio))
    val_set = set(paired[:val_count])
    LOGGER.info("Splitting USRT (%s) into %s train / %s val", len(paired), len(paired) - val_count, val_count)

    for img_path in tqdm(paired, desc="USRT split"):
        label_path = train_lbl / f"{img_path.stem}.txt"
        if img_path in val_set:
            shutil.move(str(img_path), val_img / img_path.name)
            shutil.move(str(label_path), val_lbl / label_path.name)


def assemble_dataset(project_root: Path, val_ratio: float, seed: int) -> None:
    final_img_train = project_root / "dataset/images/train"
    final_lbl_train = project_root / "dataset/labels/train"
    final_img_val = project_root / "dataset/images/val"
    final_lbl_val = project_root / "dataset/labels/val"

    for path in (final_img_train, final_lbl_train, final_img_val, final_lbl_val):
        ensure_clean_dir(path)

    mgd_root = project_root / "dataset_mgd"
    usrt_root = project_root / "dataset_usrt"

    if not mgd_root.exists():
        LOGGER.warning("MGD dataset not found at %s", mgd_root)
    else:
        train_count = copy_split(mgd_root / "images/train", mgd_root / "labels/train", final_img_train, final_lbl_train)
        val_count = copy_split(mgd_root / "images/val", mgd_root / "labels/val", final_img_val, final_lbl_val)
        LOGGER.info("Copied %s MGD train and %s MGD val samples", train_count, val_count)

    if not usrt_root.exists():
        LOGGER.warning("USRT dataset not found at %s", usrt_root)
        return

    split_usrt(usrt_root, val_ratio, seed)

    train_count = copy_split(usrt_root / "images/train", usrt_root / "labels/train", final_img_train, final_lbl_train)
    val_count = copy_split(usrt_root / "images/val", usrt_root / "labels/val", final_img_val, final_lbl_val)
    LOGGER.info("Merged %s USRT train and %s USRT val samples", train_count, val_count)
    LOGGER.info("Final dataset ready at %s", project_root / "dataset")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    assemble_dataset(args.root, args.val_ratio, args.seed)


if __name__ == "__main__":
    main()
