import streamlit as st
import utils.utils_pages as utilpg
import utils.utils_panels as utilpn

# Page config should be called for each page
utilpg.config_page()
utilpg.select_main_menu()

with st.container(border=True):
    st.markdown(
        """
        ### Configuration
        - Select config options here.
        """
    )

    sel_config = st.pills(
        "Select Config",
        ["Input Dir", "Output Dir"],
        selection_mode="single",
        default=None,
        label_visibility="collapsed",
    )

    if sel_config == "Output Dir":
        utilpn.panel_out_dir()
