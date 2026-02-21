import streamlit as st
import cv2
import numpy as np
from app_config import IMG_SIZE

def upload_image():
    st.markdown("## 📤 Upload Brain MRI Scan")
    st.caption("Supported formats: JPG, JPEG, PNG")

    st.markdown("<br>", unsafe_allow_html=True)

    upload_col = st.columns([1, 2, 1])[1]
    with upload_col:
        uploaded = st.file_uploader(
            "Upload Image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )

    if uploaded:
        img_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        st.markdown("<br>", unsafe_allow_html=True)
        img_col = st.columns([1, 2, 1])[1]
        with img_col:
            st.image(
                img, 
                caption="Uploaded Scan", 
                width="stretch" # FIX: Replaced use_container_width
            )

        return cv2.resize(img, IMG_SIZE)

    return None