# Détection de composantes dans des plans d’architecte

## Présentation
Ce projet vise à entraîner un modèle de deep learning capable de détecter automatiquement
des composantes architecturales (portes et fenêtres) à partir de plans d’architecte.
À partir de ce dépôt, il est possible de :
- cloner et installer le projet avec toutes ses librairies
- tester le pipeline avec un échantillon de données (3 train et 2 val)
- télécharger un dataset prêt à être utilisé (117 images + annotations, 36MB) à l'aide d'un script
- exécuter le projet directement depuis un notebook présent dans /notebook

## Données

- **Entrées** :
- Plans d’architecte au format image PNG
- Taille variable (redimensionnement automatique)

- **Annotations/labels** :
- Masques vectoriels au format **SVG** avec classes "Door" et "Window"

---
## Modèle

- Architecture : **U-Net**
- Encodeur : **ResNet34**
- Framework : TensorFlow / Keras
- Taille des images d’entrée : **640 × 640**
- Sortie : masque de segmentation binaire (portes + fenêtres) et fichier JSON comptant les portes et fenêtres détectées dans chaque image de validation

## Entraînement

- Fonction de perte :
- Dice + Entropie croisée
- Métriques :
- IoU (Intersection over Union)
- Coefficient de Dice
- Augmentations de données (Albumentations)

## Résultats
Les images ci-dessous montrent :
- l’image d’entrée
- la vérité terrain (annotation)
- la prédiction du modèle

*(Prédictions effectuées sur le jeu de validation)*

[Exemple avec des traits épais](assets/output_pred_ex_1.png)

[Exemple avec des traits fins](assets/output_pred_ex_2.png)


## Prérequis
- Python **3.10** ou **3.11** (64-bit recommandé)
- Environnement virtuel conseillé :
``` bash
python -m venv .env
source .env/bin/activate   # Linux / macOS
.env\Scripts\activate      # Windows
```

## Installation :
Cloner le dépot puis installer le projet en mode éditable : 
```bash
pip install -e .
```

Dépendances principales :
- tensorflow>=2.15,<2.17
- segmentation-models==1.0.1
- scipy
- requests
- numpy
- opencv-python
- albumentations
- matplotlib
- setuptools


## Utilisation :
### Préparation du dataset
Téléchargement et organisation du dataset : 
```bash
python scripts/prepare_dataset.py
```

Ou utiliser ses propres données en respectant la structure : 
```bash
données/
 ├─ images/
 │   ├─ train/
 │   └─ val/
 └─ labels/
     ├─ train/
     └─ val/
```

Ou utiliser l'échantillon présent dans sample.

### Entraînement 
Il est possible d'utiliser un échantillon de données pour vérifier le pipeline :
```bash
python src/train.py --dataset sample
```
ou d'utiliser un dataset complet :
```bash
python src/train.py
```

Pendant l'entraînement, les prédictions sont enregistrées dans le dossier outputs/callbacks sous forme d'images.
Le meilleur model est sauvegardé dans outputs/checkpoints.

### Evaluation
Évaluer l'échantillon de données : 
```bash
python src/evaluate.py --dataset sample
```
ou un dataset complet :
```bash
python src/train.py
```

Le fichier JSON de validation est enregistré dans outputs/validation.

## Limitations & améliorations possibles
- Détection des murs, indépendamment de leur épaisseur
- Support de nouveaux formats d’annotations (JSON, XML, masques PNG)
- Vectorisation des prédictions
- Évaluation orientée objets plutôt que strictement basés sur les pixels
