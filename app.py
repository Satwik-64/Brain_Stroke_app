import os
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import streamlit as st
import numpy as np
import tensorflow as tf

tf.get_logger().setLevel(logging.ERROR)


from app_config import MODEL_PATHS, MODEL_META
from core.inference import predict_stroke
from core.model_loader import load_xai_for_model
from ui.layout import top_navbar
from ui.upload import upload_image
from ui.results import show_results
from ui.model_report import render_model_report
from ui.about import render_about_page
from assets.css import inject_css

st.set_page_config(
    page_title="Brain Stroke AI",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown(inject_css(), unsafe_allow_html=True)

# State Init
if "page" not in st.session_state:
    st.session_state.page = "Detection"
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "VGG19" # Default to VGG19

# Nav
if st.session_state.page != "Model_Report":
    top_navbar()

# --- PAGE: DETECTION ---
if st.session_state.page == "Detection":
    with st.sidebar:
        st.title("⚙️ Model Settings")
        
        # Select Model
        model_options = ["VGG19", "ResNet50", "EfficientNetB0"]
        selected = st.selectbox(
            "Select AI Architecture",
            options=model_options,
            index=model_options.index(st.session_state.selected_model),
            help=f"Accuracy: {MODEL_META[st.session_state.selected_model]['accuracy']}"
        )
        
        if selected != st.session_state.selected_model:
            st.session_state.selected_model = selected
            st.rerun()

        st.success(f"**Active:** {st.session_state.selected_model}")
        st.markdown("---")
        
        if st.button("📄 View Full Model Report", use_container_width=True):
            st.session_state.page = "Model_Report"
            st.rerun()

    st.markdown(f"## 🩺 Stroke Detection: {st.session_state.selected_model}")

    @st.cache_resource(show_spinner=False)
    def load_resources(name):
        try:
            model = tf.keras.models.load_model(MODEL_PATHS[name])
            xai = load_xai_for_model(model, name)
            return model, xai
        except Exception as e:
            st.error(f"Failed to load {name}: {e}")
            return None, None

    model, xai = load_resources(st.session_state.selected_model)
    image = upload_image()

    st.markdown("<br>", unsafe_allow_html=True)
    col = st.columns([1, 3, 1])[1]
    with col:
        run_btn = st.button("🧠 Analyze Scan", use_container_width=True, disabled=(image is None))

    if image is not None and run_btn and model:
        with st.spinner("Running analysis..."):
            conf, is_stroke = predict_stroke(model, image, st.session_state.selected_model)
            label = "Stroke" if is_stroke else "Normal"
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            if not is_stroke:
                show_results(image, 1-conf, label)
            else:
                batch = np.expand_dims(image, axis=0)
                heatmap, mask = xai.generate(batch)
                show_results(image, conf, label, heatmap, mask)

elif st.session_state.page == "Model_Report":
    def go_back():
        st.session_state.page = "Detection"
        st.rerun()
    render_model_report(st.session_state.selected_model, go_back)

elif st.session_state.page == "About":
    render_about_page()