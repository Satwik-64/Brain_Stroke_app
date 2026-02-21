import tensorflow as tf
from app_config import MODEL_LAYERS
from core.xai import StrokeDetectorXAI

@tf.keras.utils.register_keras_serializable()
def f1_score(y_true, y_pred):
    import keras.backend as K
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

def load_xai_for_model(model, model_name):
    target_layer = MODEL_LAYERS.get(model_name)
    if not target_layer:
        raise ValueError(f"No XAI target layer defined for {model_name}")

    xai = StrokeDetectorXAI(
        model=model, 
        last_conv_layer_name=target_layer,
        model_name=model_name
    )
    
    return xai