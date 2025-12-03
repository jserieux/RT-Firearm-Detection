"""Convert Monash Guns Dataset VOC annotations into YOLOv8 format."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

from tqdm import tqdm


LOGGER = logging.getLogger("convert_mgd")

# Shared label space across datasets. Extend this dict if new labels appear.
CLASS_NAME_TO_ID = {
    "pistol": 0,
    "handgun": 0,
    "gun": 0,
    "weapon": 0,
    "firearm": 0,
    "short_rifle": 1,
    "rifle": 1,
}


def voc_box_to_yolo(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    img_w: float,
    img_h: float,
) -> tuple[float, float, float, float]:
    """Convert corner box (VOC) to YOLO normalized cx, cy, w, h."""
    x_center = ((xmin + xmax) / 2.0) / img_w
    y_center = ((ymin + ymax) / 2.0) / img_h
    width = (xmax - xmin) / img_w
    height = (ymax - ymin) / img_h
    return x_center, y_center, width, height


def locate_dataset_dirs(src_root: Path) -> tuple[Path, Path]:
    """Return paths to Annotations/ and JPEGImages/ under the provided root."""
    annotations_dir = src_root / "Annotations"
    images_dir = src_root / "JPEGImages"
    if annotations_dir.exists() and images_dir.exists():
        return annotations_dir, images_dir

    for child in src_root.iterdir():
        if not child.is_dir():
            continue
        cand_ann = child / "Annotations"
        cand_img = child / "JPEGImages"
        if cand_ann.exists() and cand_img.exists():
            LOGGER.info("Detected dataset under %s", child)
            return cand_ann, cand_img

    raise FileNotFoundError(
        f"Could not locate 'Annotations' and 'JPEGImages' under {src_root}. "
        "Ensure the MGD dataset is extracted or provide --src pointing to the folder containing those directories."
    )


def convert_dataset(src_root: Path, dst_root: Path) -> None:
    annotations_dir, images_dir = locate_dataset_dirs(src_root)

    dst_images = dst_root / "dataset/images/train"
    dst_labels = dst_root / "dataset/labels/train"
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    xml_files = sorted(annotations_dir.glob("*.xml"))
    if not xml_files:
        LOGGER.warning("No annotation files found in %s", annotations_dir)
        return

    processed = 0
    label_count = 0

    for xml_path in tqdm(xml_files, desc="MGD annotations"):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        filename_elem = root.find("filename")
        if filename_elem is None:
            LOGGER.warning("Missing filename in %s", xml_path.name)
            continue
        image_name = filename_elem.text
        if not image_name:
            LOGGER.warning("Empty filename tag in %s", xml_path.name)
            continue

        image_path = images_dir / image_name
        if not image_path.exists():
            LOGGER.warning("Image %s referenced in %s is missing", image_name, xml_path.name)
            continue

        size_elem = root.find("size")
        if size_elem is None:
            LOGGER.warning("Missing size info in %s", xml_path.name)
            continue
        try:
            width = float(size_elem.findtext("width"))
            height = float(size_elem.findtext("height"))
        except (TypeError, ValueError):
            LOGGER.warning("Invalid size in %s", xml_path.name)
            continue
        if width <= 0 or height <= 0:
            LOGGER.warning("Non positive dimensions in %s", xml_path.name)
            continue

        label_lines: list[str] = []
        for obj in root.findall("object"):
            raw_name = obj.findtext("name", default="").strip().lower()
            if raw_name not in CLASS_NAME_TO_ID:
                continue
            class_id = CLASS_NAME_TO_ID[raw_name]
            bbox = obj.find("bndbox")
            if bbox is None:
                continue
            try:
                xmin = float(bbox.findtext("xmin"))
                ymin = float(bbox.findtext("ymin"))
                xmax = float(bbox.findtext("xmax"))
                ymax = float(bbox.findtext("ymax"))
            except (TypeError, ValueError):
                LOGGER.warning("Invalid bbox in %s", xml_path.name)
                continue
            if xmax <= xmin or ymax <= ymin:
                continue

            x_c, y_c, w, h = voc_box_to_yolo(xmin, ymin, xmax, ymax, width, height)
            if not (0 <= x_c <= 1 and 0 <= y_c <= 1):
                continue
            if w <= 0 or h <= 0:
                continue
            label_lines.append(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")

        if not label_lines:
            continue

        dst_image_path = dst_images / image_path.name
        dst_label_path = dst_labels / (image_path.stem + ".txt")

        shutil.copy2(image_path, dst_image_path)
        with dst_label_path.open("w", encoding="utf-8") as label_file:
            label_file.writelines(label_lines)

        processed += 1
        label_count += len(label_lines)

    LOGGER.info("Finished converting MGD: %s files with %s labels", processed, label_count)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src",
        type=Path,
        default=Path("/home/dev2/Desktop/CAP 5415 Project/data/MGD"),
        help="Path to the MGD dataset root",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root where dataset/ exists",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    convert_dataset(args.src, args.dst)


if __name__ == "__main__":
    main()
