import tensorflow as tf

def iou(y_true, y_pred, smooth=1e-6):
    y_true = y_true[..., 1:]
    y_pred = y_pred[..., 1:]

    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true + y_pred) - intersection
    return (intersection + smooth) / (union + smooth)

def iou_loss(y_true, y_pred):
    return 1 - iou(y_true, y_pred)
