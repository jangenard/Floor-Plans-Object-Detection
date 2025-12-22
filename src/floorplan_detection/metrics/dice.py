import tensorflow as tf

def dice(y_true, y_pred, smooth=1e-6):
    y_true = y_true[..., 1:]
    y_pred = y_pred[..., 1:]

    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    return (2. * intersection + smooth) / (union + smooth)

def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1, 3])
    y_pred_f = tf.reshape(y_pred, [-1, 3])
    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=0)
    denom = tf.reduce_sum(y_true_f + y_pred_f, axis=0)
    dice = (2. * intersection + smooth) / (denom + smooth)
    return 1 - tf.reduce_mean(dice)

ce = tf.keras.losses.CategoricalCrossentropy()

def dice_ce_loss(y_true, y_pred):
    return 0.5 * ce(y_true, y_pred) + 0.5 * dice_loss(y_true, y_pred)
