import os

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


"""
Classe de callback qui permet de visualiser les prédictions comparées aux vrais masques pendant l'entraiment pour évaluer l'apprentissage
"""
class PredictionCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_gen, interval=1, num_samples=1, out_dir="outputs/callbacks"):
        super().__init__()
        self.val_gen = val_gen
        self.interval = interval
        self.num_samples = num_samples
        self.out_dir = out_dir

        os.makedirs(self.out_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.interval != 0:
            return
    
        fig = plt.figure(figsize=(12, 4 * self.num_samples))
    
        batch_idx = np.random.randint(len(self.val_gen))
        images, masks = self.val_gen[batch_idx]

        num_to_show = min(self.num_samples, images.shape[0])
        for i in range(num_to_show):
            img = images[i]
            gt_mask = tf.argmax(masks[i], axis=-1).numpy()
    
            pred = self.model.predict(img[None, ...], verbose=0)[0]

            conf = np.max(pred, axis=-1)
            pred_mask = np.argmax(pred, axis=-1)
            pred_mask[conf < 0.3] = 0
    
            img_vis = (img * 255).astype(np.uint8)
    
            gt_overlay = img_vis.copy()
            gt_overlay[gt_mask == 1] = [255, 0, 0]
            gt_overlay[gt_mask == 2] = [0, 255, 0]
    
            pred_overlay = img_vis.copy()
            pred_overlay[pred_mask == 1] = [255, 0, 0]
            pred_overlay[pred_mask == 2] = [0, 255, 0]
    
            plt.subplot(self.num_samples, 3, 3*i + 1)
            plt.imshow(img_vis)
            plt.title("Image")
            plt.axis("off")
    
            plt.subplot(self.num_samples, 3, 3*i + 2)
            plt.imshow(gt_overlay)
            plt.title("Masque")
            plt.axis("off")
    
            plt.subplot(self.num_samples, 3, 3*i + 3)
            plt.imshow(pred_overlay)
            plt.title("Prediction")
            plt.axis("off")
            
    
        plt.tight_layout()
        save_path = os.path.join(self.out_dir, f"epoch_{epoch:03d}.png")
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"Prédiction sauvegardée dans {self.out_dir}")