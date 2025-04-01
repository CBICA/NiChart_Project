import streamlit as st
import utils.utils_pages as utilpg

# Page config should be called for each page
utilpg.config_page()
utilpg.select_main_menu()

with st.container(border=True):
    st.markdown(
        """
        ### View session state variables
        """
    )

    sel_ssvar = st.pills(
        "Select Session State Var",
        st.session_state.keys(),
        selection_mode="single",
        default=None,
        label_visibility="collapsed",
    )
    if sel_ssvar is not None:
        st.write(st.session_state[sel_ssvar])
