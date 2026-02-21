import streamlit as st
import cv2
import numpy as np
from core.autoencoder import load_autoencoder, run_anomaly_detection

def render_about_page():
    # --- HEADER ---
    st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <h1 style="color:white; margin-bottom:0;">🧠 BrainStroke AI <span style="color:#4F8BF9">Labs</span></h1>
        <p style="color:#8b949e; font-size:1.1em;">
            Bridging the gap between clinical diagnostics and deep learning research.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # --- TABS ---
    tab1, tab2, tab3 = st.tabs(["🚀 Future Lab (Prototype)", "🔮 R&D Roadmap", "ℹ️ Project Overview"])
    
    # === TAB 1: LIVE PROTOTYPE (Autoencoder) ===
    with tab1:
        st.markdown("""
        <div style="background-color:#161b22; padding:20px; border-radius:10px; border-left:4px solid #8b949e; margin-bottom:20px;">
            <h3 style="margin-top:0;">🧪 Live Prototype: Unsupervised Anomaly Detection</h3>
            <p style="color:#b0b3b8;">
                <b>The Concept:</b> Unlike standard classifiers that look for specific patterns (Stroke vs Normal), 
                this Autoencoder is trained <i>only</i> on healthy brains. It learns to reconstruct healthy anatomy perfectly. 
                When it sees a stroke, it fails to reconstruct that specific lesion area, creating a "difference map" that highlights the anomaly.
            </p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.info("**Status:** Alpha Build (v0.1)")
            st.markdown("**Architecture:** Convolutional Autoencoder (No Skip Connections)")
            
            # Interactive Toggle
            launch_lab = st.toggle("Initialize Research Module", value=False)

        with col2:
            if not launch_lab:
                st.markdown("Toggle to load the experimental model into memory.")
            else:
                st.markdown("##### 📤 Upload MRI for Anomaly Reconstruction")
                uploaded = st.file_uploader("Select MRI Image", type=["jpg", "png"], label_visibility="collapsed")
                
                if uploaded:
                    # Convert file
                    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
                    img = cv2.imdecode(file_bytes, 1)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Run Analysis
                    model = load_autoencoder()
                    
                    if model:
                        with st.spinner("Reconstructing anatomy from latent space..."):
                            orig, recon, heatmap, final, detected = run_anomaly_detection(img, model)
                        
                        # --- RESULTS DISPLAY ---
                        st.markdown("---")
                        st.markdown(f"**AI Diagnosis:** {'🔴 ANOMALY DETECTED' if detected else '🟢 HEALTHY STRUCTURE'}")
                        
                        r1, r2, r3, r4 = st.columns(4)
                        
                        with r1:
                            st.image(orig, caption="Input Scan", use_container_width=True)
                        with r2:
                            st.image(recon, caption="Reconstruction (Healthy)", use_container_width=True)
                        with r3:
                            # Colorize heatmap
                            h_norm = (heatmap / np.max(heatmap) * 255).astype(np.uint8)
                            h_color = cv2.applyColorMap(h_norm, cv2.COLORMAP_JET)
                            st.image(h_color, caption="Error Map (Residuals)", use_container_width=True)
                        with r4:
                            st.image(final, caption="Segmentation Prediction", use_container_width=True)

    # === TAB 2: THEORETICAL ROADMAP ===
    with tab2:
        st.header("Future Implementation Horizons")
        
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("#### 1. 3D Volumetric Analysis (NIfTI)")
            st.info("""
            **Theory:** Strokes are 3D volumes, not 2D slices. Processing single slices loses Z-axis context.
            \n**Implementation:** Move from 2D CNNs to **3D-ResNet** or **V-Net** architectures that ingest full DICOM/NIfTI volumes. This allows for accurate volume calculation (mL) crucial for thrombectomy decisions.
            """)
            
            st.markdown("#### 2. Federated Learning (Privacy-First)")
            st.info("""
            **Theory:** Medical data is siloed due to privacy (HIPAA/GDPR).
            \n**Implementation:** Deploy a **Federated Learning** pipeline where the model travels to the hospital's data, trains locally, and only sends weight updates back to the central server. No patient data ever leaves the secure facility.
            """)

        with c2:
            st.markdown("#### 3. Multi-Modal Fusion")
            st.info("""
            **Theory:** Diagnosis relies on more than just images (e.g., patient age, blood pressure, time since onset).
            \n**Implementation:** Construct a **Dual-Stream Network**. Stream A processes the MRI (CNN), Stream B processes Electronic Health Records (EHR) via an MLP. These features are concatenated before the final classification layer for holistic diagnosis.
            """)
            
            st.markdown("#### 4. Uncertainty Quantification")
            st.info("""
            **Theory:** Doctors need to know *when* the AI is unsure.
            \n**Implementation:** Implement **Monte Carlo Dropout** during inference. By running the prediction 50 times with random dropout, we can measure the variance in predictions. High variance = Low Confidence, triggering a "Human Review Required" flag.
            """)

    # === TAB 3: PROJECT OVERVIEW ===
    with tab3:
        st.markdown("""
        ### 🏥 Clinical Objective
        To develop a robust Clinical Decision Support System (CDSS) for the automated detection and localization of ischemic and hemorrhagic stroke lesions in T2-weighted MRI scans.
        
        ### 🛠 Technology Stack
        
        **Deep Learning Frameworks**
        - **TensorFlow/Keras:** Core model development and training.
        - **OpenCV:** Image preprocessing, contour detection, and morphological operations.
        
        **Model Architectures**
        - **VGG19:** Utilized for its depth and texture extraction capabilities.
        - **ResNet50:** Employed for residual learning to prevent gradient degradation.
        - **EfficientNetB0:** Optimized for high-efficiency, low-latency inference.
        - **Autoencoder (Experimental):** Unsupervised reconstruction network for anomaly detection.
        
        **Explainable AI (XAI)**
        - **Grad-CAM:** Gradient-weighted Class Activation Mapping for visual explainability.
        - **Reconstruction Error:** Pixel-wise difference mapping for unsupervised segmentation.
        
        **Deployment**
        - **Streamlit:** Interactive web interface for real-time inference.
        - **Python:** Primary development language.
        
        ---
        <div style="text-align:center; color:grey; font-size:0.8em;">
            BrainStroke AI v1.0 | Research & Development Edition
        </div>
        """, unsafe_allow_html=True)