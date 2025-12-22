import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2

from floorplan_detection.data.svg_to_mask import svg_to_mask
from floorplan_detection.data.resize import resize_with_padding

"""
Classe dans laquelle a lieu la boucle de pré-traitement des images et des masques
"""
class FloorplanGenerator(tf.keras.utils.Sequence):
    def __init__(self, pairs, batch_size=4, augment=None, shuffle=True, **kwargs):
        super().__init__(**kwargs)
        self.pairs = pairs
        self.batch_size = batch_size
        self.augment = augment
        self.shuffle = shuffle
        self.indices = np.arange(len(pairs))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.pairs) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, idx):
        batch_idx = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
    
        images = []
        masks = []
    
        for i in batch_idx:
            img_path, svg_path = self.pairs[i]
    
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            orig_h, orig_w = img.shape[:2]

            mask = svg_to_mask(svg_path, out_shape=(orig_h, orig_w))

            img, scale, (pad_y, pad_x) = resize_with_padding(img, target=640)

            mask = cv2.resize(
                mask,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_NEAREST
            )

            mask = cv2.copyMakeBorder(
                mask,
                pad_y, 640 - mask.shape[0] - pad_y,
                pad_x, 640 - mask.shape[1] - pad_x,
                cv2.BORDER_CONSTANT,
                value=0
            )
            if self.augment:
                augmented = self.augment(image=img, mask=mask)
                img = augmented["image"]
                mask = augmented["mask"]
    
            images.append(img)
            masks.append(mask)
    
        images = np.array(images, dtype=np.float32) / 255.0
        masks = tf.one_hot(np.array(masks), depth=3)
    
        return images, masks