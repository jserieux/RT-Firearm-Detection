"""Convert USRT dataset annotations (COCO or VOC) into YOLO format."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

from tqdm import tqdm

LOGGER = logging.getLogger("convert_usrt")

CLASS_NAME_TO_ID = {
    "pistol": 0,
    "handgun": 0,
    "gun": 0,
    "weapon": 0,
    "firearm": 0,
    "short_rifle": 1,
    "rifle": 1,
}


def ensure_ext(name: str) -> str:
    if name.lower().endswith(('.jpg', '.jpeg', '.png')):
        return name
    return f"{name}.jpg"


def convert_voc(xml_files: list[Path], images_dir: Path, dst_images: Path, dst_labels: Path) -> tuple[int, int]:
    processed = 0
    labels = 0
    for xml_path in tqdm(xml_files, desc="USRT VOC annotations"):
        try:
            tree = ET.parse(xml_path)
        except ET.ParseError:
            LOGGER.warning("Failed to parse %s", xml_path)
            continue
        root = tree.getroot()
        filename = root.findtext("filename", default="").strip()
        if not filename:
            LOGGER.warning("Missing filename in %s", xml_path)
            continue
        filename = ensure_ext(filename)
        image_path = images_dir / f"{filename}"
        if not image_path.exists():
            image_path = xml_path.with_suffix(".jpg")
            if not image_path.exists():
                LOGGER.warning("Image missing for %s", xml_path)
                continue

        width = float(root.findtext("size/width", default="0"))
        height = float(root.findtext("size/height", default="0"))
        if width <= 0 or height <= 0:
            LOGGER.warning("Invalid dimensions in %s", xml_path)
            continue

        label_lines: list[str] = []
        for obj in root.findall("object"):
            name = obj.findtext("name", default="").strip().lower()
            if name not in CLASS_NAME_TO_ID:
                continue
            class_id = CLASS_NAME_TO_ID[name]
            bbox = obj.find("bndbox")
            if bbox is None:
                continue
            try:
                xmin = float(bbox.findtext("xmin"))
                ymin = float(bbox.findtext("ymin"))
                xmax = float(bbox.findtext("xmax"))
                ymax = float(bbox.findtext("ymax"))
            except (TypeError, ValueError):
                continue
            if xmax <= xmin or ymax <= ymin:
                continue
            x_c = ((xmin + xmax) / 2.0) / width
            y_c = ((ymin + ymax) / 2.0) / height
            w = (xmax - xmin) / width
            h = (ymax - ymin) / height
            if w <= 0 or h <= 0:
                continue
            label_lines.append(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")

        if not label_lines:
            continue

        dst_img = dst_images / Path(filename).name
        dst_lbl = dst_labels / (Path(filename).stem + ".txt")
        shutil.copy2(image_path, dst_img)
        with dst_lbl.open("w", encoding="utf-8") as f:
            f.writelines(label_lines)
        processed += 1
        labels += len(label_lines)
    return processed, labels


def convert_coco(json_path: Path, images_dir: Path, dst_images: Path, dst_labels: Path) -> tuple[int, int]:
    data = json.loads(json_path.read_text())
    images = {img["id"]: img for img in data.get("images", [])}
    cat_to_id = {}
    for cat in data.get("categories", []):
        name = cat["name"].strip().lower()
        if name in CLASS_NAME_TO_ID:
            cat_to_id[cat["id"]] = CLASS_NAME_TO_ID[name]
    processed_images: set[int] = set()
    label_cache: dict[int, list[str]] = {}
    for ann in data.get("annotations", []):
        img_id = ann["image_id"]
        if ann.get("category_id") not in cat_to_id:
            continue
        img_info = images.get(img_id)
        if not img_info:
            continue
        w, h = float(img_info["width"]), float(img_info["height"])
        if w <= 0 or h <= 0:
            continue
        x, y, bw, bh = ann["bbox"]
        x_c = (x + bw / 2.0) / w
        y_c = (y + bh / 2.0) / h
        ww = bw / w
        hh = bh / h
        if ww <= 0 or hh <= 0:
            continue
        class_id = cat_to_id[ann["category_id"]]
        label_cache.setdefault(img_id, []).append(f"{class_id} {x_c:.6f} {y_c:.6f} {ww:.6f} {hh:.6f}\n")

    processed = 0
    labels = 0
    for img_id, label_lines in tqdm(label_cache.items(), desc="USRT COCO annotations"):
        img_info = images[img_id]
        img_name = img_info["file_name"]
        src_img = images_dir / img_name
        if not src_img.exists():
            LOGGER.warning("Missing image %s", src_img)
            continue
        dst_img = dst_images / src_img.name
        dst_lbl = dst_labels / (src_img.stem + ".txt")
        shutil.copy2(src_img, dst_img)
        with dst_lbl.open("w", encoding="utf-8") as f:
            f.writelines(label_lines)
        processed += 1
        labels += len(label_lines)
        processed_images.add(img_id)
    return processed, labels


def convert_dataset(src_root: Path, dst_root: Path) -> None:
    dst_images = dst_root / "dataset_usrt/images/train"
    dst_labels = dst_root / "dataset_usrt/labels/train"
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    json_path = src_root / "annotations.json"
    images_dir = src_root / "Images"
    if json_path.exists():
        processed, labels = convert_coco(json_path, images_dir, dst_images, dst_labels)
    else:
        voc_files = sorted(images_dir.glob("*.xml"))
        if not voc_files:
            raise FileNotFoundError("No annotations found for USRT dataset")
        processed, labels = convert_voc(voc_files, images_dir, dst_images, dst_labels)
    LOGGER.info("Converted USRT: %s labeled images, %s boxes", processed, labels)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src",
        type=Path,
        default=Path("/home/dev2/Desktop/CAP 5415 Project/data/USRT 2FPS"),
        help="Path to USRT dataset root",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root containing dataset/",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    convert_dataset(args.src, args.dst)


if __name__ == "__main__":
    main()
