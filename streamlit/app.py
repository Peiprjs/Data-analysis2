import streamlit as st
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'notebooks'))

# Lazy import page modules - only import when actually needed
# This improves initial page load performance
def get_page_module(page_name):
    """Lazy load page modules on demand to improve performance"""
    if page_name == "Introduction":
        from page_modules import introduction
        return introduction
    elif page_name == "FAIRness":
        from page_modules import fair_compliance
        return fair_compliance
    elif page_name == "Exploratory Data Analysis":
        from page_modules import eda
        return eda
    elif page_name == "Models Overview":
        from page_modules import models_overview
        return models_overview
    elif page_name == "Model Training":
        from page_modules import models
        return models
    elif page_name == "Model Interpretability":
        from page_modules import interpretability
        return interpretability
    elif page_name == "Conclusions":
        from page_modules import conclusions
        return conclusions
    else:
        raise ValueError(f"Unknown page: {page_name}")

# Debug logging
print("=" * 80)
print("DEBUG: app.py starting execution")
print(f"DEBUG: Python version: {sys.version}")
print(f"DEBUG: Current working directory: {os.getcwd()}")
print("=" * 80)

st.set_page_config(
    page_title='Microbiome Data Analysis - LucKi Cohort',
    page_icon=None,
    initial_sidebar_state=570,
    layout='wide',
    menu_items={
        'Get Help': 'https://github.com/MAI-David/Data-analysis/issues',
        'Report a bug': 'https://github.com/MAI-David/Data-analysis/issues',
        'About': """
        # LucKi Microbiome Analysis for Age Prediction 
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

    [data-testid="stAppViewContainer"] .main {
        background: #111827;
    }

    @media (prefers-color-scheme: dark) {
        [data-testid="stAppViewContainer"] .main {
            background: #161f2d;
        }
    }

    @media (prefers-color-scheme: light) {
        [data-testid="stAppViewContainer"] .main {
            background: #e9edf3;
        }
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
        background-color: #0f172a;
        color: #e2e8f0;
    }

    [data-testid="stSidebar"] * {
        color: #e2e8f0;
    }

    @media (prefers-color-scheme: light) {
        [data-testid="stSidebar"] {
            background-color: #e3f2fd;
            color: #1565C0;
        }
        
        [data-testid="stSidebar"] * {
            color: #1565C0;
        }
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

# Define available pages (names only for performance)
PAGES = [
    "Introduction",
    "FAIRness",
    "Exploratory Data Analysis",
    "Models Overview",
    "Model Training",
    "Model Interpretability",
    "Conclusions"
]

print("=" * 80)
print("DEBUG: Available pages:")
for page_name in PAGES:
    print(f"  - {page_name}")
print("=" * 80)
st.sidebar.html("""
    <div style="text-align: center; padding: -1rem 0;">
        <h1 style="font-size: 1.5rem; margin: 0; color: #2E7D32;">LucKi Microbiome Analysis for Age Prediction</h1>
    </div>
    """)
st.sidebar.markdown("---")
st.sidebar.markdown("### Navigation")
print("DEBUG: About to create sidebar radio widget")
print(f"DEBUG: Available pages: {PAGES}")
print(f"DEBUG: Default selection index: {PAGES.index('Introduction') if 'Introduction' in PAGES else 0}")
selection = st.sidebar.radio(
    "Select a page",
    PAGES,
    index=PAGES.index("Introduction") if "Introduction" in PAGES else 0,
    key="page_selection"
)

print("=" * 80)
print(f"DEBUG: User selected page: '{selection}'")
print(f"DEBUG: Selection type: {type(selection)}")
print(f"DEBUG: Selected page is in PAGES list: {selection in PAGES}")
print("=" * 80)
st.sidebar.html("""
    <style>
    [data-testid="stSidebar"] .stRadio > div[role="radiogroup"] {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
        width: 100%;
        box-sizing: border-box;
    }

    [data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label[data-baseweb="radio"] > div:first-child {
        display: none !important;
    }

    [data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label[data-baseweb="radio"] {
        display: block;
        width: 100%;
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }

    [data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label[data-baseweb="radio"] > div {
        display: flex;
        width: 110%;
        box-sizing: border-box;
        justify-content: center;
        align-items: center;
        border-radius: 0.75rem;
        padding: 0.6rem 1.1rem 0.6rem 1.4rem;
        border: 1px solid rgba(255,255,255,0.15);
        background: linear-gradient(135deg, rgba(30,41,59,0.9), rgba(15,23,42,0.95));
        color: #e2e8f0;
        cursor: pointer;
        transition: background-color 0.15s ease, color 0.15s ease, border-color 0.15s ease, box-shadow 0.15s ease;
        position: relative;
        font-weight: 600;
    }

    [data-testid="stSidebar"] .stRadio input[type="radio"] {
        display: none;
    }

    [data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label[data-baseweb="radio"] > div:hover {
        border-color: rgba(46,125,50,0.35);
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    }

    [data-testid="stSidebar"] .stRadio input[type="radio"]:checked + div {
        background: linear-gradient(135deg, #2E7D32, #1B5E20);
        color: white;
        border-color: #2E7D32;
        box-shadow: inset 0 0 0 1px rgba(0,0,0,0.08), 0 6px 14px rgba(46,125,50,0.35);
    }

    [data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label[data-baseweb="radio"] > div::before {
        content: "";
        position: absolute;
        left: 0.75rem;
        width: 6px;
        height: 70%;
        border-radius: 999px;
        background: rgba(255,255,255,0.12);
        transition: background-color 0.15s ease, height 0.15s ease;
    }

    [data-testid="stSidebar"] .stRadio input[type="radio"]:checked + div::before {
        background: #c9f7d4;
        height: 80%;
    }

    @media (prefers-color-scheme: light) {
        [data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label[data-baseweb="radio"] > div {
            border: 1px solid rgba(21,101,192,0.2);
            background: linear-gradient(135deg, rgba(227,242,253,0.5), rgba(187,222,251,0.6));
            color: #1565C0;
        }

        [data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label[data-baseweb="radio"] > div:hover {
            border-color: rgba(46,125,50,0.4);
            box-shadow: 0 2px 8px rgba(46,125,50,0.15);
        }

        [data-testid="stSidebar"] .stRadio input[type="radio"]:checked + div {
            background: linear-gradient(135deg, #2E7D32, #1B5E20);
            color: white;
            border-color: #2E7D32;
            box-shadow: inset 0 0 0 1px rgba(0,0,0,0.08), 0 6px 14px rgba(46,125,50,0.35);
        }

        [data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label[data-baseweb="radio"] > div::before {
            background: rgba(21,101,192,0.2);
        }

        [data-testid="stSidebar"] .stRadio input[type="radio"]:checked + div::before {
            background: #c9f7d4;
        }
    }

    </style>
""")
print("DEBUG: Custom sidebar styling applied")
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
        st.markdown("""
            <style>
            .main .block-container, .main .block-container p, .main .block-container div, 
            .main .block-container span, .main .block-container li {
                font-size: 1.1rem !important;
            }
            </style>
        """, unsafe_allow_html=True)
    elif font_size == "Extra Large":
        st.markdown("""
            <style>
            .main .block-container, .main .block-container p, .main .block-container div, 
            .main .block-container span, .main .block-container li {
                font-size: 1.2rem !important;
            }
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

# Render the selected page with lazy loading
print("=" * 80)
print(f"DEBUG: About to render page: '{selection}'")
try:
    # Lazy load the page module only when needed
    page = get_page_module(selection)
    print(f"DEBUG: Lazy-loaded page module: {page}")
    print(f"DEBUG: Page module has 'app' attribute: {hasattr(page, 'app')}")
    
    if hasattr(page, 'app'):
        print(f"DEBUG: Calling {selection}.app() function...")
        page.app()
        print(f"DEBUG: Successfully rendered page: '{selection}'")
    else:
        print(f"ERROR: Page module '{selection}' does not have an 'app()' function!")
        st.error(f"Page '{selection}' is not properly configured. Missing app() function.")
except Exception as e:
    print(f"ERROR: Failed to render page '{selection}': {str(e)}")
    import traceback
    traceback.print_exc()
    st.error(f"Error loading page '{selection}': {str(e)}")
print("=" * 80)
