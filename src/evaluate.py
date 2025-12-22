import glob
import os 
import json
import warnings
import argparse
from pathlib import Path
os.environ["SM_FRAMEWORK"] = "tf.keras"

#Pour supprimer divers warnings pas utiles
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import cv2
from scipy import ndimage
import numpy as np
import tensorflow as tf

from floorplan_detection.metrics.iou import iou, iou_loss
from floorplan_detection.metrics.dice import dice, dice_ce_loss

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["sample", "full"],
        default="full",
        help="Choisir la taille des données de validation"
    )
    return parser.parse_args()


def count_objects(mask):
    counts = {}
    for cls, name in [(1, "Door"), (2, "Window")]:
        _, num = ndimage.label(mask == cls)
        counts[name] = int(num)
    return counts

def main() :
    """
    Évaluation du modèle, on pourra prendre des images d'autres datasets pour estimer la capacité de généralisation du modèle
    L'évaluation renvoie un fichier JSON où sont comptés les portes et les fenêtres pour chaque image
    """
    args = parse_args()

    if args.dataset == "sample":
        val_folder = "./sample/images/val"
    else:
        val_folder = "./data/images/val"

    results = {}

    MODEL_PATH = "models/unet.keras"

    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            "dice_ce_loss": dice_ce_loss,
            "iou": iou,
            "dice": dice
        }
    )   

    for img_path in sorted(glob.glob(os.path.join(val_folder, "*.png"))):
        image = cv2.imread(img_path)
        image = cv2.resize(image, (640, 640))
        image = image / 255.0

        pred = model.predict(image[None, ...], verbose=0)[0]

        conf = np.max(pred, axis=-1)
        cls = np.argmax(pred, axis=-1)
        cls[conf < 0.5] = 0

        counts = count_objects(cls)

        img_name = os.path.basename(img_path)
        results[img_name] = counts

    """
    On enregistre les prédictions sur les données de validation dans un json pour avoir une trace écrite du nombre de portes et de fenêtres dans chaque image
    """
    output_dir = Path("./outputs/validation")
    output_dir.mkdir(parents=True, exist_ok=True)

    file_name = f"eval_predictions_{args.dataset}.json"

    output_path = output_dir / file_name

    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print(
        f"Predictions sauvegardées pour toutes les images du dossier "
        f"{val_folder} dans le fichier {output_path}"
    )

if __name__ == "__main__":
    main()