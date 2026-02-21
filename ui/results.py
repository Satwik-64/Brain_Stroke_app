import streamlit as st
import cv2
import numpy as np

def show_results(image, confidence, predicted_label, heatmap=None, mask=None):

    st.markdown("---")
    st.markdown(
        f"""
        <h2 style='text-align:center;'>
            Prediction:
            <span style='color:{"#ff4b4b" if predicted_label=="Stroke" else "#22c55e"}'>
                {predicted_label}
            </span>
        </h2>
        <p style='text-align:center; font-size:18px;'>
            Confidence: <b>{confidence:.2%}</b>
        </p>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # NORMAL CASE
    if predicted_label == "Normal":
        img_col = st.columns([1, 2, 1])[1]
        with img_col:
            st.image(
                image,
                caption="Original Scan (No Stroke Detected)",
                width="stretch"  # FIX: Replaced use_container_width
            )
        return

    # STROKE CASE
    col1, col2, col3 = st.columns([1, 1, 1])

    # 1. Original
    with col1:
        st.image(image, caption="Original Scan", width="stretch") # FIX

    # 2. Grad-CAM
    with col2:
        if heatmap is not None:
            # Ensure proper scaling for CV2 ColorMap
            heatmap_norm = np.uint8(255 * heatmap) if heatmap.max() <= 1.0 else heatmap.astype(np.uint8)
            heatmap_resized = cv2.resize(heatmap_norm, (image.shape[1], image.shape[0]))
            
            heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
            heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
            
            img_uint8 = np.uint8(image)
            overlay = cv2.addWeighted(img_uint8, 0.6, heatmap_color, 0.4, 0)
            
            st.image(overlay, caption="Grad-CAM Activation (Red=High)", width="stretch") # FIX

    # 3. Segmentation
    with col3:
        if mask is not None:
            seg = np.array(image).copy() # Safe copy
            
            # Resize mask
            if mask.shape[:2] != seg.shape[:2]:
                mask = cv2.resize(mask, (seg.shape[1], seg.shape[0]))

            # Yellow Overlay
            yellow = np.zeros_like(seg)
            yellow[mask == 255] = [255, 255, 0]
            seg = cv2.addWeighted(seg, 1.0, yellow, 0.4, 0)

            # Bounding Box
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                c = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(c)
                
                # Small padding
                pad = 5
                x, y = max(0, x - pad), max(0, y - pad)
                w, h = w + 2 * pad, h + 2 * pad

                cv2.rectangle(seg, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(seg, "Lesion", (x, max(y - 5, 10)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            st.image(seg, caption="Lesion Localization", width="stretch") # FIX