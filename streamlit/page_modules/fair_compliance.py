import streamlit as st


def app():
    st.title("FAIRness")
    st.markdown(
        """
        This study follows the **FAIR principles** to ensure data and models remain
        findable, accessible, interoperable, and reusable.

        - **Findable:** Clear naming, stable repository links, and versioned releases.
        - **Accessible:** Open-source code and documentation with public issue tracking.
        - **Interoperable:** Standard CSV/tabular formats and taxonomic naming conventions.
        - **Reusable:** Documented preprocessing (encoding, CLR), genus-level feature set, and cached splits.
        """
    )
    
    st.markdown("---")
    
    st.subheader("AI Guidelines Compliance")
    st.markdown(
        """
        This project adheres to established AI reporting guidelines and best practices.
        Detailed compliance checklists are maintained in the `checklists/` folder:
        
        - **TRIPOD+AI**: Transparent Reporting of a multivariable prediction model for 
          Individual Prognosis Or Diagnosis + Artificial Intelligence
          - Location: `checklists/TRIPOD_AI_COMPLIANCE.md`
          - Status: 78% fully compliant, 18% partially compliant
          
        - **DECIDE-AI**: A checklist for reporting of evaluation studies in medical AI
          - Location: `checklists/DECIDE_AI_COMPLIANCE.md`
          - Ensures transparent reporting of study design, data sources, and statistical methods
        
        These checklists ensure:
        - Transparent model development and validation procedures
        - Clear documentation of AI methods and algorithms
        - Reproducible research with fixed random seeds and version control
        - Proper reporting of model performance and limitations
        - Ethical considerations and bias assessment
        """
    )