import cv2

"""
Redimension des images
"""
def resize_with_padding(img, target=640):
    h, w = img.shape[:2]
    scale = target / max(h, w)
    nh, nw = int(h * scale), int(w * scale)

    img = cv2.resize(img, (nw, nh))
    pad_h = target - nh
    pad_w = target - nw

    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    img = cv2.copyMakeBorder(
        img, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=0
    )

    return img, scale, (top, left)

