import os
import warnings
import argparse

os.environ["SM_FRAMEWORK"] = "tf.keras"

#Pour supprimer divers warnings peu utiles
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import tensorflow as tf

from floorplan_detection.data.dataset import FloorplanGenerator
from floorplan_detection.data.transforms import get_train_transforms
from floorplan_detection.data.collect_pairs import collect_pairs
from floorplan_detection.models.unet import build_unet
from floorplan_detection.callbacks.prediction_callback import PredictionCallback
from floorplan_detection.metrics.iou import iou, iou_loss
from floorplan_detection.metrics.dice import dice, dice_ce_loss

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["sample", "full"],
        default="full",
        help="Choisir la taille des données"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    model = build_unet()
    train_augment = get_train_transforms()

    if args.dataset == "sample":
        train_pairs = collect_pairs(
            img_dir="sample/images/train",
            svg_dir="sample/labels/train"
        )

        val_pairs = collect_pairs(
            img_dir="sample/images/val",
            svg_dir="sample/labels/val"
        )
        epochs = 3
    else:
        train_pairs = collect_pairs(
            img_dir="data/images/train",
            svg_dir="data/labels/train"
        )
        val_pairs = collect_pairs(
            img_dir="data/images/val",
            svg_dir="data/labels/val"
        )
        epochs=100

    """
    Chargement des données pré-traitées avec l'augmentation
    """
    train_gen = FloorplanGenerator(
        train_pairs,
        batch_size=4,
        augment=train_augment
    )

    val_gen = FloorplanGenerator(
        val_pairs,
        batch_size=4,
        augment=None,
        shuffle=False
    )

    """
    Callback pour visualiser une prédiction sur num_samples données de validation chaque interval epoch
    """
    callback = PredictionCallback(
        val_gen=val_gen,
        interval=1,
        num_samples=2
    )

    """
    Arrêter l'apprentissage si la métrique monitor (ici val_loss) ne s'améliore pas après patience epochs
    """
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_iou",
        patience=8,
        restore_best_weights=True,
        mode='max'
    )

    """
    Checkpoint pour sauvegarder le meilleur modèle et le meilleur résultat
    """
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath="outputs/checkpoints/best_model.keras",
        monitor="val_iou",
        save_best_only=True,
        save_weights_only=False,
        mode='max'
    )

    """
    Compilation du modèle avec l'optimizer Adam, la loss iou loss et les métriques
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), 
        loss=dice_ce_loss,
        metrics=[
            dice,
            iou
        ]
    )

    MODEL_DIR = "saved_models"
    MODEL_PATH = os.path.join(MODEL_DIR, "unet.keras")

    os.makedirs(MODEL_DIR, exist_ok=True)

    """
    Entrainement du modèle sur les données train pré-traitées
    """
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=[early_stop, checkpoint, callback]
    )
    
    """
    Sauvegarde du model dans MODEL_PATH pour le réutiliser pour la validation
    """
    model.save(MODEL_PATH)

if __name__ == "__main__":
    main()