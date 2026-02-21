import streamlit as st

def top_navbar():
    st.markdown("""
    <style>
    .nav-container {
        display: flex;
        justify-content: center;
        background-color: #0e1117;
        padding: 10px;
        margin-bottom: 20px;
        border-bottom: 1px solid #262730;
    }
    .nav-btn {
        margin: 0 15px;
        text-decoration: none;
        font-size: 18px;
        font-weight: 500;
        cursor: pointer;
    }
    </style>
    """, unsafe_allow_html=True)

    # We use Streamlit columns to simulate a nav bar
    # 2 Main Options: Detection | About
    col1, col2, col3 = st.columns([4, 1, 1])
    
    with col1:
        st.markdown("### 🧠 Brain Stroke AI")
        
    with col2:
        if st.button("🔍 Detection", use_container_width=True):
            st.session_state.page = "Detection"
            st.rerun()

    with col3:
        if st.button("ℹ️ About", use_container_width=True):
            st.session_state.page = "About"
            st.rerun()

    st.markdown("---")