import streamlit as st
import utils.utils_pages as utilpg
import os



# Page config should be called for each page
utilpg.config_page()
utilpg.show_menu()

st.markdown(
    """
    ### View Results 
    - Select options to view/report results here.
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
        disp_session_state()
if sel_config == "Output Files":
    with st.container(border=True):
        st.markdown('##### '+ st.session_state.navig['task'] + ':')
        disp_folder_tree()

