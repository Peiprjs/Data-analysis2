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