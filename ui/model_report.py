import streamlit as st
import os
from config import MODEL_META, CONFUSION_MATRICES

def render_model_report(model_name, go_back_callback):
    """
    Renders an industry-standard 'Model Card' for the selected AI.
    Focuses on Transparency, Clinical Metrics, and Intended Use.
    """
    meta = MODEL_META.get(model_name, {})
    cm_path = CONFUSION_MATRICES.get(model_name, "")
    
    # --- HEADER ---
    col1, col2 = st.columns([1, 6])
    with col1:
        if st.button("← Back"):
            go_back_callback()
    with col2:
        st.title(f"🛡️ Model Card: {model_name}")
        st.caption(f"**Architecture:** {meta.get('architecture')} | **Type:** {meta.get('type')}")

    st.markdown("---")

    # --- SECTION 1: CLINICAL PERFORMANCE METRICS ---
    st.subheader("🏥 Clinical Performance (Test Set)")
    
    # Using columns for key metrics
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    with kpi1:
        st.metric(
            "Sensitivity (Recall)", 
            meta.get("sensitivity"), 
            help="True Positive Rate. The ability to correctly identify actual stroke cases."
        )
    
    with kpi2:
        st.metric(
            "Specificity", 
            meta.get("specificity"), 
            help="True Negative Rate. The ability to correctly identify normal cases."
        )
        
    with kpi3:
        st.metric(
            "Accuracy", 
            meta.get("accuracy"),
            help="Overall correctness of the model predictions."
        )
        
    with kpi4:
        st.metric(
            "ROC-AUC", 
            meta.get("auc"),
            help="Area Under Receiver Operating Characteristic Curve. 1.0 is perfect discrimination."
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # --- SECTION 2: DATA & TRANSPARENCY ---
    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown("#### 📂 Training Data & Methodology")
        st.info(f"**Dataset:** {meta.get('dataset_info')}")
        
        st.markdown("#### ✅ Intended Use Case")
        st.success(meta.get("intended_use"))

    with c2:
        st.markdown("#### ⚠️ Known Limitations")
        st.warning(meta.get("limitations"))
        
        st.markdown("#### 📉 Confusion Matrix")
        # Visual placeholder for the confusion matrix
        if os.path.exists(cm_path):
            st.image(cm_path, caption=f"{model_name} Test Set Confusion Matrix", use_container_width=True)
        else:
            # Professional placeholder if image is missing
            st.markdown(
                f"""
                <div style="
                    border: 1px dashed #4c4c4c; 
                    background-color: #262730;
                    padding: 30px; 
                    text-align: center; 
                    border-radius: 8px;">
                    <p style="color: #fafafa; margin:0;"><b>Confusion Matrix Visualization</b></p>
                    <p style="color: #9ca0a6; font-size: 0.9em; margin-top:5px;">
                        (Image asset <code>{model_name}_cm.png</code> not found)
                    </p>
                </div>
                """, 
                unsafe_allow_html=True
            )

    st.markdown("---")

    # --- SECTION 3: COMPLIANCE FOOTER ---
    st.markdown("""
    <div style="font-size: 0.85em; color: #808495;">
        <b>Compliance Statement:</b> This model is intended for research and investigational use only. 
        It has not been cleared or approved by the FDA or any other regulatory authority for clinical diagnosis. 
        Results should always be verified by a qualified radiologist.
    </div>
    """, unsafe_allow_html=True)