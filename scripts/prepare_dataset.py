import os
import zipfile
import shutil
from pathlib import Path
import random

import requests

DATA_URL = "http://dag.cvc.uab.es/DATASETS/CVC-FP.zip"
DATA_DIR = Path("data")
ZIP_PATH = DATA_DIR / "dataset.zip"
EXTRACT_DIR = DATA_DIR / "CVC-FP" / "ImagesGT"

TRAIN_RATIO = 0.8

def download_data(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        print(f"{dest} existe déjà, téléchargement annulé")
        return 0

    print(f"Téléchargement du dataset depuis {url}")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(dest, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print("Téléchargement fini")
    return 1

def unzip(zip_path, extract_to):
    print(f"Extraction {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    print("Extraction complète")
    os.remove(zip_path)

"""
Trouver les paires images/annotations dans le dataset à partir d'une liste.
Autorise aussi le formatage des conventions de noms.
En l'occurence, les annotations sont nommées {image}_gt_{nombre}, cette fonction permet d'ignorer ce qui se trouve après le "_gt_". 
"""
def collect_pairs_from_lists(images, labels, suffix="_gt_"):

    label_map = {}
    for label_path in labels:
        stem = label_path.stem
        if suffix in stem:
            prefix = stem.split(suffix)[0]
        else:
            prefix = stem
        label_map[prefix] = label_path

    pairs = []
    for img_path in images:
        img_stem = img_path.stem
        if img_stem in label_map:
            pairs.append((img_path, label_map[img_stem]))
        else:
            print(f"Pas d'annotation trouvée pour {img_stem}")

    return pairs

def organize_dataset(extract_dir, dest_dir, train_ratio=0.8):
    """
    Sépare les images et les labels et crée des dossiers respectifs
    """
    all_files = list(extract_dir.rglob("*"))
    images = [f for f in all_files if f.suffix.lower() == ".png"]
    labels = [f for f in all_files if f.suffix.lower() == ".svg"]

    pairs = collect_pairs_from_lists(images, labels, suffix="_gt_")
    print(f"{len(pairs)} paires image-label trouvées")

    pairs = list(pairs)
    random.shuffle(pairs)
    split_idx = int(len(pairs) * train_ratio)
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]

    for split in ["train", "val"]:
        (dest_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dest_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    for img_path, lbl_path in train_pairs:
        shutil.copy(img_path, dest_dir / "images/train")
        shutil.copy(lbl_path, dest_dir / "labels/train")

    for img_path, lbl_path in val_pairs:
        shutil.copy(img_path, dest_dir / "images/val")
        shutil.copy(lbl_path, dest_dir / "labels/val")

def main():
    if download_data(DATA_URL, ZIP_PATH) : 
        unzip(ZIP_PATH, EXTRACT_DIR)
        organize_dataset(EXTRACT_DIR, DATA_DIR)
    shutil.rmtree('data/CVC-FP')

if __name__ == "__main__":
    main()