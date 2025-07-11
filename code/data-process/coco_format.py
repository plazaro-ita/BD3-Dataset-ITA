"""
bd3_to_coco.py
Convert BD3 folder structure to train/val/test COCO JSONs with one
full-image box per picture.
"""
import json, os, glob
from PIL import Image
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm

ROOT = "BD3-Dataset/curated-dataset"          # folders Algae/, Major Crack/, ...
OUT  = "detectron2/datasets/coco_bd3_20250710"

os.makedirs(OUT, exist_ok=True)

cats = [{"id": i, "name": n} for i, n in enumerate(
    ["algae","major_crack","minor_crack",
     "peeling","spalling","stain","normal"], start=1)]

def make_split(img_paths, split):
    split_dir = os.path.join(OUT, "images", split)
    os.makedirs(split_dir, exist_ok=True)
    
    images, annots = [], []
    ann_id = 1
    for img_id, p in enumerate(tqdm(img_paths), start=1):
        w, h = Image.open(p).size
        cls   = os.path.basename(os.path.dirname(p))
        catId = next(c["id"] for c in cats if c["name"] == cls)
        fname = os.path.basename(p)
        
        # copy images to the directory
        dest_path = os.path.join(split_dir, fname)
        if not os.path.exists(dest_path):
            shutil.copy(p, dest_path)

        images.append({"id": img_id,
                       "file_name": f"{split}/{fname}",
                       "width": w, "height": h})
        annots.append({"id": ann_id, "image_id": img_id,
                       "category_id": catId,
                       "bbox": [0, 0, w, h],
                       "area": w*h, "iscrowd": 0})
        ann_id += 1
    json_path = f"{OUT}/annotations/{split}.json"
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    json.dump(
        {"images": images, "annotations": annots, "categories": cats},
        open(json_path, "w"), indent=2)

all_imgs = sorted(glob.glob(f"{ROOT}/**/*.jpg", recursive=True))
train, val = train_test_split(all_imgs, test_size=0.10, random_state=42, stratify=[os.path.dirname(p) for p in all_imgs])

for split, paths in zip(["train","val"], [train,val]):
    make_split(paths, split)
