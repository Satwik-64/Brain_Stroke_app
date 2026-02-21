import tensorflow as tf
import numpy as np
import cv2
from core.preprocessing import custom_preprocessing

class StrokeDetectorXAI:
    def __init__(self, model, last_conv_layer_name, model_name="ResNet50"):
        self.model = model
        self.last_conv_layer_name = last_conv_layer_name
        self.model_name = model_name
        
        # Handle Keras List Output
        output_node = model.output
        if isinstance(output_node, (list, tuple)):
            output_node = output_node[0]

        # Grad-CAM Model
        self.grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(last_conv_layer_name).output, output_node]
        )

    def generate(self, img_batch):
        """
        img_batch: Raw input images (N, 224, 224, 3)
        """
        # 1. Preprocessing
        img_preprocessed = np.array([
            custom_preprocessing(img, self.model_name) for img in img_batch
        ])

        # 2. Gradient Tape
        with tf.GradientTape() as tape:
            outputs = self.grad_model(img_preprocessed)
            last_conv_layer_output = outputs[0]
            preds = outputs[1]

            if isinstance(preds, (list, tuple)):
                preds = preds[0]
            
            class_channel = preds[:, 0]
        
        # 3. Compute Gradients
        grads = tape.gradient(class_channel, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # 4. Generate Heatmap
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # ReLU & Normalize
        heatmap = tf.maximum(heatmap, 0)
        max_val = tf.math.reduce_max(heatmap)
        if max_val == 0:
            heatmap = heatmap
        else:
            heatmap = heatmap / (max_val + 1e-10)
            
        heatmap = heatmap.numpy()

        # 5. Generate Segmentation Mask (IMPROVED LOGIC)
        
        # A. Resize first
        heatmap_resized = cv2.resize(heatmap, (224, 224))
        
        # B. Gaussian Blur to smooth the 'red' peak into the 'yellow' surrounding
        # This helps capture the full lesion shape instead of just the tiny red center
        heatmap_blur = cv2.GaussianBlur(heatmap_resized, (11, 11), 0)
        
        # C. Convert to uint8
        heatmap_uint8 = np.uint8(255 * heatmap_blur)
        
        # D. Thresholding (0.35 / 35%)
        # 35% captures Red + Orange + Yellow (Warm regions)
        # But safely ignores Green/Blue (Background)
        thresh_val = int(255 * 0.35)
        _, mask = cv2.threshold(
            heatmap_uint8, thresh_val, 255, cv2.THRESH_BINARY
        )
        
        # E. Keep ONLY the Largest Contour (Main Lesion)
        # This prevents the box from becoming "very big" by ignoring disjoint noise dots.
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        clean_mask = np.zeros_like(mask)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            # Only draw if it's a significant size
            if cv2.contourArea(largest_contour) > 20: 
                cv2.drawContours(clean_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        
        # Final cleanup on the clean mask
        kernel = np.ones((5, 5), np.uint8)
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)

        return heatmap_resized, clean_mask