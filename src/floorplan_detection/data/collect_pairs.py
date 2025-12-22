import glob
import os

from pathlib import Path

def collect_pairs(img_dir, svg_dir):
    pairs = []

    svg_map = {}
    for svg_path in glob.glob(os.path.join(svg_dir, "*.svg")):
        name = Path(svg_path).stem
        if "_gt_" in name:
            prefix = name.split("_gt_")[0]
        else:
            prefix = name
        svg_map[prefix] = svg_path

    for img_path in glob.glob(os.path.join(img_dir, "*.png")):
        img_stem = Path(img_path).stem

        if img_stem in svg_map:
            pairs.append((img_path, svg_map[img_stem]))
        else:
            print("No SVG for:", img_stem)

    return pairs