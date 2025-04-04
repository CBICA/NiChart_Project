import streamlit as st
import utils.utils_pages as utilpg

# Page config should be called for each page
utilpg.config_page()
utilpg.select_main_menu()

st.markdown(
    """
    ### Debugging Options (Dev)
    - Select debugging options here.
    """
)

sel_config = st.pills(
    "Select Debug",
    ["Session State", "Output Files"],
    selection_mode="single",
    default=None,
    label_visibility="collapsed",
)

if sel_config == "Session State":
    with st.container(border=True):
        sel_ssvars = st.pills(
            "Select Session State Variable(s) to View",
            st.session_state.keys(),
            selection_mode="multi",
            default=None,
            #label_visibility="collapsed",
        )
        for sel_var in sel_ssvars:
            st.write(sel_var)
            st.write(st.session_state[sel_var])

if sel_config == "Output Files":
    st.write('Work in progress ...')

