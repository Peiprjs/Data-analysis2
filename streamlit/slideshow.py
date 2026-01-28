import streamlit as st
from streamlit.components.v1 import html
import time

st.set_page_config(
    page_title='Microbiome Data Analysis - Presentation',
    page_icon='🧬',
    layout='wide',
    initial_sidebar_state='collapsed'
)

st.markdown("""
    <style>
    .main .block-container {
        padding: 2rem;
        max-width: 100%;
    }
    
    .slide {
        min-height: 80vh;
        display: flex;
        flex-direction: column;
        justify-content: center;
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .slide h1 {
        font-size: 3.5rem !important;
        font-weight: 700 !important;
        margin-bottom: 1rem !important;
        color: white !important;
        text-align: center;
    }
    
    .slide h2 {
        font-size: 2.5rem !important;
        font-weight: 600 !important;
        margin-bottom: 1.5rem !important;
        color: white !important;
    }
    
    .slide p, .slide li {
        font-size: 1.3rem !important;
        line-height: 1.8 !important;
        color: white !important;
    }
    
    .slide-content {
        background: rgba(255, 255, 255, 0.1);
        padding: 2rem;
        border-radius: 10px;
        backdrop-filter: blur(10px);
    }
    
    .slide-number {
        position: absolute;
        bottom: 2rem;
        right: 2rem;
        font-size: 1.2rem;
        opacity: 0.7;
    }
    
    .metric-box {
        background: rgba(255, 255, 255, 0.2);
        padding: 1.5rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem;
    }
    
    .metric-value {
        font-size: 3rem;
        font-weight: 700;
        color: #FFD700;
    }
    
    .metric-label {
        font-size: 1.2rem;
        margin-top: 0.5rem;
        opacity: 0.9;
    }
    
    .stButton > button {
        font-size: 1.2rem;
        padding: 1rem 2rem;
        background-color: #FFD700;
        color: #333;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)

slides = []

slide_1 = """
<div class="slide" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
    <div style="text-align: center;">
        <h1>🧬 Microbiome Data Analysis Platform</h1>
        <h2>Machine Learning for the LucKi Cohort</h2>
        <p style="font-size: 1.5rem; margin-top: 2rem; opacity: 0.9;">
            Predicting age groups from gut microbiome profiles
        </p>
        <p style="font-size: 1.2rem; margin-top: 3rem; opacity: 0.8;">
            Data Analysis Team | 2024
        </p>
    </div>
    <div class="slide-number">Slide 1</div>
</div>
"""

slide_2 = """
<div class="slide" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
    <h2>📊 Dataset Overview</h2>
    <div class="slide-content">
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-top: 2rem;">
            <div class="metric-box">
                <div class="metric-value">930</div>
                <div class="metric-label">Stool Samples</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">~6,900</div>
                <div class="metric-label">Microbiome Features</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">80/20</div>
                <div class="metric-label">Train/Test Split</div>
            </div>
        </div>
        <ul style="margin-top: 2rem; font-size: 1.3rem;">
            <li>MetaPhlAn 4.1.1 taxonomic profiling</li>
            <li>High-dimensional sparse data (~80% zeros)</li>
            <li>Multiple age groups across different families</li>
            <li>Log-normal abundance distribution</li>
        </ul>
    </div>
    <div class="slide-number">Slide 2</div>
</div>
"""

slide_3 = """
<div class="slide" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
    <h2>🔬 Data Preprocessing Pipeline</h2>
    <div class="slide-content">
        <div style="font-size: 1.4rem; line-height: 2.5;">
            <p><strong>1. Data Integration</strong></p>
            <p style="margin-left: 2rem;">→ Merge abundance tables with metadata</p>
            
            <p style="margin-top: 1rem;"><strong>2. Label Encoding</strong></p>
            <p style="margin-left: 2rem;">→ Convert categorical variables (family, sex, age group)</p>
            
            <p style="margin-top: 1rem;"><strong>3. Quality Control</strong></p>
            <p style="margin-left: 2rem;">→ Missing value detection, outlier analysis</p>
            
            <p style="margin-top: 1rem;"><strong>4. CLR Transformation</strong></p>
            <p style="margin-left: 2rem;">→ Handle compositional nature of microbiome data</p>
            
            <p style="margin-top: 1rem;"><strong>5. Feature Selection</strong></p>
            <p style="margin-left: 2rem;">→ Filter to genus-level features</p>
        </div>
    </div>
    <div class="slide-number">Slide 3</div>
</div>
"""

slide_4 = """
<div class="slide" style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);">
    <h2>🤖 Machine Learning Models</h2>
    <div class="slide-content">
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 2rem; margin-top: 2rem;">
            <div>
                <h3 style="color: white; font-size: 1.8rem;">Ensemble Methods</h3>
                <ul style="font-size: 1.2rem;">
                    <li>Random Forest</li>
                    <li>Gradient Boosting</li>
                    <li>AdaBoost</li>
                </ul>
            </div>
            <div>
                <h3 style="color: white; font-size: 1.8rem;">Advanced Methods</h3>
                <ul style="font-size: 1.2rem;">
                    <li>XGBoost</li>
                    <li>LightGBM</li>
                    <li>Neural Networks</li>
                </ul>
            </div>
        </div>
        <div style="margin-top: 2rem; background: rgba(255,255,255,0.2); padding: 1.5rem; border-radius: 8px;">
            <p style="font-size: 1.3rem; text-align: center;">
                <strong>Evaluation Metrics:</strong> RMSE, R² Score, MAE, Cross-Validation
            </p>
        </div>
    </div>
    <div class="slide-number">Slide 4</div>
</div>
"""

slide_5 = """
<div class="slide" style="background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); color: #333;">
    <h2 style="color: #333;">🔍 Model Interpretability</h2>
    <div class="slide-content" style="background: rgba(255, 255, 255, 0.3);">
        <div style="font-size: 1.3rem;">
            <div style="margin-bottom: 2rem;">
                <h3 style="color: #333; font-size: 1.8rem;">Feature Importance</h3>
                <p style="color: #333;">Identify which microbes contribute most to predictions</p>
            </div>
            
            <div style="margin-bottom: 2rem;">
                <h3 style="color: #333; font-size: 1.8rem;">LIME Explanations</h3>
                <p style="color: #333;">Local interpretable model-agnostic explanations for individual samples</p>
            </div>
            
            <div>
                <h3 style="color: #333; font-size: 1.8rem;">SHAP Values</h3>
                <p style="color: #333;">SHapley Additive exPlanations based on game theory</p>
            </div>
        </div>
        
        <div style="margin-top: 2rem; background: rgba(0,0,0,0.1); padding: 1rem; border-radius: 8px; text-align: center;">
            <p style="font-size: 1.2rem; color: #333;">
                <strong>Goal:</strong> Make black-box models transparent and trustworthy
            </p>
        </div>
    </div>
    <div class="slide-number" style="color: #333;">Slide 5</div>
</div>
"""

slide_6 = """
<div class="slide" style="background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%); color: #333;">
    <h2 style="color: #333;">📊 Key Results</h2>
    <div class="slide-content" style="background: rgba(255, 255, 255, 0.3);">
        <ul style="font-size: 1.4rem; color: #333; line-height: 2.2;">
            <li><strong>Best Model:</strong> Random Forest and XGBoost show strong performance</li>
            <li><strong>Feature Selection:</strong> Genus-level features provide good balance of performance and interpretability</li>
            <li><strong>Cross-Validation:</strong> Consistent performance across folds</li>
            <li><strong>Ensemble Methods:</strong> Combining models improves robustness</li>
            <li><strong>Key Features:</strong> Specific bacterial genera strongly associated with age</li>
        </ul>
        
        <div style="margin-top: 2rem; background: rgba(0,0,0,0.1); padding: 1.5rem; border-radius: 8px; text-align: center;">
            <p style="font-size: 1.3rem; color: #333;">
                Machine learning successfully captures age-related microbiome patterns
            </p>
        </div>
    </div>
    <div class="slide-number" style="color: #333;">Slide 6</div>
</div>
"""

slide_7 = """
<div class="slide" style="background: linear-gradient(135deg, #e0c3fc 0%, #8ec5fc 100%);">
    <h2>🚀 Interactive Platform Features</h2>
    <div class="slide-content">
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1.5rem; margin-top: 2rem;">
            <div style="background: rgba(255,255,255,0.2); padding: 1.5rem; border-radius: 8px;">
                <h3 style="color: white; font-size: 1.5rem;">🏠 Home</h3>
                <p>Project overview and dataset statistics</p>
            </div>
            
            <div style="background: rgba(255,255,255,0.2); padding: 1.5rem; border-radius: 8px;">
                <h3 style="color: white; font-size: 1.5rem;">🔬 Preprocessing</h3>
                <p>Interactive data transformation</p>
            </div>
            
            <div style="background: rgba(255,255,255,0.2); padding: 1.5rem; border-radius: 8px;">
                <h3 style="color: white; font-size: 1.5rem;">🤖 Training</h3>
                <p>Train and compare ML models</p>
            </div>
            
            <div style="background: rgba(255,255,255,0.2); padding: 1.5rem; border-radius: 8px;">
                <h3 style="color: white; font-size: 1.5rem;">🔍 Interpretability</h3>
                <p>Understand model predictions</p>
            </div>
            
            <div style="background: rgba(255,255,255,0.2); padding: 1.5rem; border-radius: 8px;">
                <h3 style="color: white; font-size: 1.5rem;">📊 Results</h3>
                <p>Cross-validation and comparison</p>
            </div>
            
            <div style="background: rgba(255,255,255,0.2); padding: 1.5rem; border-radius: 8px;">
                <h3 style="color: white; font-size: 1.5rem;">♿ Accessible</h3>
                <p>WCAG compliant interface</p>
            </div>
        </div>
    </div>
    <div class="slide-number">Slide 7</div>
</div>
"""

slide_8 = """
<div class="slide" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
    <h2>🎯 Impact & Applications</h2>
    <div class="slide-content">
        <ul style="font-size: 1.4rem; line-height: 2.2;">
            <li><strong>Healthcare:</strong> Age-related microbiome changes inform personalized medicine</li>
            <li><strong>Research:</strong> Open-source platform for microbiome analysis</li>
            <li><strong>Reproducibility:</strong> FAIR principles ensure scientific rigor</li>
            <li><strong>Education:</strong> Interactive learning tool for bioinformatics</li>
            <li><strong>Scalability:</strong> Can be adapted to other cohorts and research questions</li>
        </ul>
        
        <div style="margin-top: 2rem; background: rgba(255,255,255,0.2); padding: 1.5rem; border-radius: 8px; text-align: center;">
            <p style="font-size: 1.3rem;">
                <strong>Future Work:</strong> Integration with other omics data, larger cohorts, 
                longitudinal analysis
            </p>
        </div>
    </div>
    <div class="slide-number">Slide 8</div>
</div>
"""

slide_9 = """
<div class="slide" style="background: linear-gradient(135deg, #30cfd0 0%, #330867 100%);">
    <div style="text-align: center;">
        <h1>Thank You!</h1>
        <div style="margin-top: 3rem; font-size: 1.4rem;">
            <p>🧬 Microbiome Data Analysis Platform</p>
            <p style="margin-top: 1rem;">📚 github.com/Peiprjs/Data-analysis2</p>
            <p style="margin-top: 1rem;">📧 Contact via GitHub Issues</p>
        </div>
        
        <div style="margin-top: 4rem; font-size: 1.2rem; opacity: 0.8;">
            <p><strong>Acknowledgments:</strong></p>
            <p style="margin-top: 1rem;">LucKi Cohort Participants • MetaPhlAn Team</p>
            <p>Open Source Community • Research Collaborators</p>
        </div>
    </div>
    <div class="slide-number">Slide 9</div>
</div>
"""

st.sidebar.markdown("### 📽️ Presentation Controls")
slide_num = st.sidebar.number_input("Go to slide", min_value=1, max_value=9, value=1, step=1)

if st.sidebar.button("◀ Previous"):
    slide_num = max(1, slide_num - 1)
    
if st.sidebar.button("Next ▶"):
    slide_num = min(9, slide_num + 1)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Current Slide:** {slide_num} / 9")

if st.sidebar.checkbox("Auto-advance slides", value=False):
    auto_delay = st.sidebar.slider("Delay (seconds)", 5, 30, 10)
    time.sleep(auto_delay)
    if slide_num < 9:
        slide_num += 1
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("### 💡 Tips")
st.sidebar.info("""
- Use number input to jump to any slide
- Use Previous/Next buttons to navigate
- Enable auto-advance for automatic slideshow
- Press F11 for fullscreen mode
""")

slides_dict = {
    1: slide_1,
    2: slide_2,
    3: slide_3,
    4: slide_4,
    5: slide_5,
    6: slide_6,
    7: slide_7,
    8: slide_8,
    9: slide_9
}

st.markdown(slides_dict[slide_num], unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    if st.button("⏮ First Slide"):
        slide_num = 1
        st.rerun()
with col2:
    st.markdown(f"<h3 style='text-align: center;'>Slide {slide_num} of 9</h3>", unsafe_allow_html=True)
with col3:
    if st.button("Last Slide ⏭"):
        slide_num = 9
        st.rerun()
