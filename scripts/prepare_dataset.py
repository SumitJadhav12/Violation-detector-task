"""
Dataset Preparation Script for Hard Hat / Helmet Safety Detection
Extracts and converts Kaggle Pascal VOC annotations to YOLO format.
Creates train/val/test splits.
"""

import os
import sys
import zipfile
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm

CLASS_MAPPING = {
    "helmet": 0,    # helmet
    "head": 1,      # no_helmet
    # 'person' is ignored to focus strictly on helmet vs no_helmet per assessment prompt
}

def convert_voc_bbox(size, box):
    """Convert VOC bbox (xmin, xmax, ymin, ymax) to YOLO format (x_center, y_center, width, height)."""
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    xmin, xmax, ymin, ymax = box
    x_center = (xmin + xmax) / 2.0
    y_center = (ymin + ymax) / 2.0
    w = xmax - xmin
    h = ymax - ymin
    return x_center * dw, y_center * dh, w * dw, h * dh

def prepare_dataset(
    zip_path: str = "outputs/uploads/archive(9).zip",
    dest_dir: str = "datasets/helmet_safety",
    split_ratios: tuple = (0.70, 0.15, 0.15),
    seed: int = 42
):
    print("============================================================")
    print(" PREPARING HELMET SAFETY DATASET (YOLO FORMAT)")
    print("============================================================")
    print(f"[*] Source Zip: {zip_path}")
    print(f"[*] Destination: {dest_dir}")

    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Source archive not found: {zip_path}")

    # Set random seed for reproducibility
    random.seed(seed)

    # Create destination directories
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(dest_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(dest_dir, "labels", split), exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as z:
        all_names = z.namelist()
        
        # Identify image and annotation pairs
        xml_files = {
            os.path.splitext(os.path.basename(f))[0]: f
            for f in all_names if f.startswith("annotations/") and f.endswith(".xml")
        }
        
        image_files = {
            os.path.splitext(os.path.basename(f))[0]: f
            for f in all_names if f.startswith("images/") and not f.endswith("/")
        }

        # Find matching base IDs
        common_ids = sorted(list(set(xml_files.keys()) & set(image_files.keys())))
        print(f"[*] Total matching image-annotation pairs: {len(common_ids)}")

        # Shuffle
        random.shuffle(common_ids)

        n_total = len(common_ids)
        n_train = int(n_total * split_ratios[0])
        n_val = int(n_total * split_ratios[1])

        train_ids = common_ids[:n_train]
        val_ids = common_ids[n_train:n_train + n_val]
        test_ids = common_ids[n_train + n_val:]

        splits = {
            "train": train_ids,
            "val": val_ids,
            "test": test_ids
        }

        print(f"    - Train split: {len(train_ids)} images")
        print(f"    - Val split:   {len(val_ids)} images")
        print(f"    - Test split:  {len(test_ids)} images")

        stats = {"helmet": 0, "no_helmet": 0, "ignored_person": 0}

        for split, id_list in splits.items():
            print(f"\n[*] Extracting and converting {split} split ({len(id_list)} images)...")
            img_dest_dir = os.path.join(dest_dir, "images", split)
            lbl_dest_dir = os.path.join(dest_dir, "labels", split)

            for sample_id in tqdm(id_list, desc=f"Writing {split}"):
                img_path_in_zip = image_files[sample_id]
                xml_path_in_zip = xml_files[sample_id]

                # Extract image
                img_ext = os.path.splitext(img_path_in_zip)[1]
                target_img_filename = f"{sample_id}{img_ext}"
                target_img_path = os.path.join(img_dest_dir, target_img_filename)

                with open(target_img_path, "wb") as f_out:
                    f_out.write(z.read(img_path_in_zip))

                # Parse XML and convert to YOLO labels
                xml_content = z.read(xml_path_in_zip)
                root = ET.fromstring(xml_content)
                size_node = root.find("size")
                if size_node is not None:
                    width = float(size_node.find("width").text)
                    height = float(size_node.find("height").text)
                else:
                    width, height = 640.0, 640.0

                if width <= 0 or height <= 0:
                    width, height = 640.0, 640.0

                yolo_lines = []
                for obj in root.findall("object"):
                    name = obj.find("name").text.strip().lower()
                    if name in CLASS_MAPPING:
                        class_id = CLASS_MAPPING[name]
                        if class_id == 0:
                            stats["helmet"] += 1
                        else:
                            stats["no_helmet"] += 1

                        bnd = obj.find("bndbox")
                        xmin = max(0.0, float(bnd.find("xmin").text))
                        ymin = max(0.0, float(bnd.find("ymin").text))
                        xmax = min(width, float(bnd.find("xmax").text))
                        ymax = min(height, float(bnd.find("ymax").text))

                        if xmax > xmin and ymax > ymin:
                            x_c, y_c, w, h = convert_voc_bbox((width, height), (xmin, xmax, ymin, ymax))
                            # Clamp values to [0, 1]
                            x_c = max(0.0, min(1.0, x_c))
                            y_c = max(0.0, min(1.0, y_c))
                            w = max(0.0, min(1.0, w))
                            h = max(0.0, min(1.0, h))
                            yolo_lines.append(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")
                    elif name == "person":
                        stats["ignored_person"] += 1

                # Write label file
                target_lbl_path = os.path.join(lbl_dest_dir, f"{sample_id}.txt")
                with open(target_lbl_path, "w", encoding="utf-8") as f_lbl:
                    f_lbl.write("\n".join(yolo_lines))

    print("\n============================================================")
    print(" DATASET EXTRACTION & CONVERSION COMPLETE")
    print("============================================================")
    print(f"[+] Total Helmet instances:    {stats['helmet']}")
    print(f"[+] Total No-Helmet instances: {stats['no_helmet']}")
    print(f"[+] Ignored person bboxes:     {stats['ignored_person']}")
    print(f"[+] Dataset saved to:          {os.path.abspath(dest_dir)}")

if __name__ == "__main__":
    prepare_dataset()
