import streamlit as st
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'notebooks'))

from pages import introduction, eda, models_overview, conclusions

st.set_page_config(
    page_title='Microbiome Data Analysis - LucKi Cohort',
    page_icon=None,
    initial_sidebar_state='expanded',
    layout='wide',
    menu_items={
        'Get Help': 'https://github.com/MAI-David/Data-analysis/issues',
        'Report a bug': 'https://github.com/MAI-David/Data-analysis/issues',
        'About': """
        # Microbiome Data Analysis Platform
        Version 1.0.0
        
        Analyze microbiome data from the LucKi cohort using machine learning.
        """
    }
)

st.markdown("""
    <style>
    * {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica', 'Arial', sans-serif;
    }
    
    button:focus, input:focus, select:focus, textarea:focus {
        outline: 3px solid #2E7D32 !important;
        outline-offset: 2px !important;
    }
    
    .main .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    
    h1 {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        line-height: 1.2 !important;
        margin-bottom: 1rem !important;
        color: var(--text-color);
    }
    
    h2 {
        font-size: 2rem !important;
        font-weight: 600 !important;
        margin-top: 1.5rem !important;
        margin-bottom: 0.75rem !important;
        color: var(--text-color);
    }
    
    h3 {
        font-size: 1.5rem !important;
        font-weight: 600 !important;
        margin-top: 1rem !important;
        margin-bottom: 0.5rem !important;
        color: var(--text-color);
    }
    
    a {
        color: #1565C0 !important;
        text-decoration: underline !important;
    }
    
    a:hover {
        color: #0D47A1 !important;
    }
    
    .stButton > button {
        background-color: #2E7D32;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: background-color 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #1B5E20;
        color: white;
    }
    
    [data-testid="stSidebar"] {
        background-color: var(--secondary-background-color);
    }
    
    [data-testid="stSidebar"] .stRadio > label {
        font-size: 1rem;
        font-weight: 500;
    }
    
    .dataframe {
        font-size: 0.9rem;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
    }
    
    .stAlert {
        border-radius: 4px;
        border-left: 4px solid;
    }
    
    .skip-to-main {
        position: absolute;
        top: -40px;
        left: 0;
        background: #2E7D32;
        color: white;
        padding: 8px;
        text-decoration: none;
        z-index: 100;
    }
    
    .skip-to-main:focus {
        top: 0;
    }
    
    @media (prefers-color-scheme: dark) {
        :root {
            --text-color: #E0E0E0;
            --background-color: #1E1E1E;
            --secondary-background-color: #2D2D2D;
        }
    }
    
    @media (max-width: 768px) {
        .main .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
        
        h1 {
            font-size: 2rem !important;
        }
        
        h2 {
            font-size: 1.5rem !important;
        }
    }
    
    .stSpinner > div {
        border-color: #2E7D32 transparent transparent transparent;
    }
    
    .stRadio > div[role="radiogroup"] > label[data-baseweb="radio"] > div:first-child {
        background-color: #2E7D32;
    }
    </style>    """, unsafe_allow_html=True)

PAGES = {
    "Introduction": introduction,
    "Exploratory Data Analysis": eda,
    "Models": models_overview,
    "Conclusions": conclusions
}

st.sidebar.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <h1 style="font-size: 1.8rem; margin: 0; color: #2E7D32;">Microbiome Analysis</h1>
        <h2 style="font-size: 1.1rem; margin: 0.5rem 0 0 0;">LucKi Cohort Platform</h2>
        <p style="font-size: 0.85rem; color: #666; margin: 0.25rem 0 0 0;">Accessible and FAIR-aligned</p>
    </div>
    """, unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### Navigation")

selection = st.sidebar.segmented_control(
    "Select a page",
    list(PAGES.keys()),
    default="Introduction",
    width="stretch"
)

st.sidebar.markdown("---")

with st.sidebar.expander("About", expanded=False):
    st.markdown("""
    **Version:** 1.0.0
    
    **Purpose:** Interactive platform for analyzing microbiome data from the LucKi cohort 
    using machine learning models.
    
    **Dataset:** 930 samples, ~6,900 features
    
    **Models:** Random Forest, XGBoost, Gradient Boosting, LightGBM, Neural Networks
    """)

with st.sidebar.expander("Settings", expanded=False):
    st.markdown("**Accessibility Options**")
    
    font_size = st.selectbox(
        "Font Size",
        ["Normal", "Large", "Extra Large"],
        help="Adjust text size for better readability"
    )
    
    if font_size == "Large":
        st.markdown("<style>body { font-size: 1.1rem; }</style>", unsafe_allow_html=True)
    elif font_size == "Extra Large":
        st.markdown("<style>body { font-size: 1.2rem; }</style>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("**Theme**")
    theme_choice = st.selectbox(
        "Color Mode",
        ["Light", "Dark"],
        help="Switch between light and dark mode"
    )
    
    # Apply theme-specific styling
    if theme_choice == "Dark":
        bg_color = "#0E1117"
        secondary_bg = "#262730"
        text_color = "#FAFAFA"
    else:
        bg_color = "#FFFFFF"
        secondary_bg = "#F5F5F5"
        text_color = "#1E1E1E"
    
    st.markdown(f"""
        <style>
        :root {{
            --background-color: {bg_color} !important;
            --secondary-background-color: {secondary_bg} !important;
            --text-color: {text_color} !important;
        }}
        .stApp {{
            background-color: {bg_color} !important;
        }}
        .main {{
            background-color: {bg_color} !important;
        }}
        [data-testid="stSidebar"] {{
            background-color: {secondary_bg} !important;
        }}
        h1, h2, h3, h4, h5, h6, p, span, div, label {{
            color: {text_color} !important;
        }}
        .stMarkdown {{
            color: {text_color} !important;
        }}
        </style>
    """, unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("""
    <div style="text-align: center; font-size: 0.8rem; color: #666;">
        <p><a href="https://github.com/MAI-David/Data-analysis" target="_blank">Documentation</a></p>
        <p><a href="https://github.com/MAI-David/Data-analysis/issues" target="_blank">Report Issues</a></p>
        <p style="margin-top: 1rem;">© 2026 Team David</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div id="main-content"></div>', unsafe_allow_html=True)

page = PAGES[selection]
page.app()
