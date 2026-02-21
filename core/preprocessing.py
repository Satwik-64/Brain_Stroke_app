import cv2
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess

def custom_preprocessing(image, model_name="ResNet50"):
    """
    Applies noise reduction + Model-specific preprocessing.
    Args:
        image: RGB image (numpy array), usually uint8.
        model_name: "ResNet50", "VGG19", or "EfficientNetB0".
    """
    # 1. Ensure image format
    if image.dtype != 'uint8':
        image = image.astype('uint8')

    # 2. Apply Bilateral Filter (Noise Reduction) - Common step
    # Keeps edges sharp while removing noise
    filtered_img = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

    # 3. Convert to float32 for Keras
    img_float = filtered_img.astype('float32')

    # 4. Apply Model-Specific Logic
    if model_name == "VGG19":
        # VGG19 (Keras) expects BGR and Zero-Centering
        return vgg_preprocess(img_float)
    
    elif model_name == "EfficientNetB0":
        # EfficientNet expects 0-255 inputs (it handles scaling internally usually)
        # or specific EfficientNet scaling depending on version
        return effnet_preprocess(img_float)
    
    else:
        # Default (ResNet50) expects Zero-Centering (Caffe style)
        return resnet_preprocess(img_float)