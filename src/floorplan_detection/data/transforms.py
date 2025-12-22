import cv2
import albumentations as A

"""
Augmentation des données d'entrainement avec rotations verticales, horizontales, flou gaussien et diverses transformations affines
"""
def get_train_transforms():
    train_augment = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Affine(
            translate_percent=0.02,
            scale=(0.95, 1.05),          
            rotate=(-2, 2),              
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST,
            p=0.5
        ),
        A.GaussianBlur(blur_limit=3, p=0.2)
    ])
    return train_augment