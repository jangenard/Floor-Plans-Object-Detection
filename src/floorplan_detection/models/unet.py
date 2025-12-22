from segmentation_models import Unet

def build_unet():
    """
    Création du modèle avec la dimension de nos images, le nombre de classe à prédire et la fonction d'activation
    """
    model = Unet('resnet34', 
        input_shape=(640,640,3), 
        classes=3, 
        activation='softmax'
    )

    return model