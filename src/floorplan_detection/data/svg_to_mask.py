import numpy as np
import cv2
import xml.etree.ElementTree as ET

"""
Définition des classes à prédire, dans notre cas "Other" correspond à la majorité des images, on l'ignora pendant l'apprentissage
"""
CLASS_MAP = {
    "Other": 0,
    "Door": 1,
    "Window" : 2
}

"""
Conversion des annotations svg vers des masques numpy avec redimension pour correspondre aux images
"""
def svg_to_mask(svg_path, out_shape):
    tree = ET.parse(svg_path)
    root = tree.getroot()

    def find_tag(tag):
        return root.find(f".//{{*}}{tag}")

    width_el = find_tag("width")
    height_el = find_tag("height")

    if width_el is None or height_el is None:
        raise ValueError(f"SVG missing width/height: {svg_path}")

    svg_w = float(width_el.text)
    svg_h = float(height_el.text)

    H, W = out_shape

    scale_x = W / svg_w
    scale_y = H / svg_h

    mask = np.zeros((H, W), dtype=np.uint8)

    for el in root.iter():
        tag = el.tag.split("}")[-1]
        cls = el.attrib.get("class")
        if cls not in CLASS_MAP:
            continue

        cid = CLASS_MAP[cls]

        if tag == "polygon":
            pts = []
            for p in el.attrib["points"].split():
                x, y = map(float, p.split(","))
                pts.append([int(x * scale_x), int(y * scale_y)])

            pts = np.array(pts, np.int32)
            cv2.fillPoly(mask, [pts], cid)

    return mask