import tensorflow as tf
import numpy as np
import cv2
from config import MODEL_PATHS
import streamlit as st

@st.cache_resource(show_spinner=False)
def load_autoencoder():
    try:
        return tf.keras.models.load_model(MODEL_PATHS["Autoencoder"])
    except Exception as e:
        st.error(f"Autoencoder model not found: {e}")
        return None

def run_anomaly_detection(image, model):
    """
    Runs the Reconstruction-based Anomaly Detection pipeline.
    """
    # 1. Preprocess (Resize & Normalize)
    target_size = (224, 224)
    if image.shape[:2] != target_size:
        original_img = cv2.resize(image, target_size)
    else:
        original_img = image.copy()
        
    # Normalize [0,1]
    input_tensor = original_img.astype(np.float32) / 255.0
    input_tensor = np.expand_dims(input_tensor, axis=0)

    # 2. Inference (Reconstruct Healthy Version)
    reconstructed_tensor = model.predict(input_tensor, verbose=0)
    reconstructed_img = reconstructed_tensor[0] 
    # Scale back to 0-255 for display
    reconstructed_display = (reconstructed_img * 255).astype(np.uint8)

    # 3. Error Calculation (The "Heatmap")
    diff = np.abs(input_tensor[0] - reconstructed_img)
    heatmap = np.mean(diff, axis=-1)

    # 4. Segmentation Logic
    # Blur to merge noise
    heatmap_blurred = cv2.GaussianBlur(heatmap, (9, 9), 0)
    
    # Dynamic Threshold (Mean + 3 Std Devs)
    threshold = np.mean(heatmap_blurred) + 3.0 * np.std(heatmap_blurred)
    binary_mask = (heatmap_blurred > threshold).astype(np.uint8)
    
    # Morphological Closing (Fuse blobs)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    
    # 5. Contours & Overlay
    final_output = original_img.copy()
    contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    stroke_detected = False
    
    if contours:
        # Get largest blob
        largest_cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_cnt)
        
        # Filter tiny noise
        if area > 150:
            stroke_detected = True
            x, y, w, h = cv2.boundingRect(largest_cnt)
            
            # Red BBox
            cv2.rectangle(final_output, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(final_output, "ANOMALY", (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # Red Overlay
            red_layer = np.zeros_like(final_output)
            red_layer[:, :, 0] = closed_mask * 255 
            final_output = cv2.addWeighted(final_output, 1.0, red_layer, 0.4, 0)

    return original_img, reconstructed_display, heatmap_blurred, final_output, stroke_detected