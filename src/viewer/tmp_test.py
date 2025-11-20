import argparse
import os

import streamlit as st
import utils.utils_pages as utilpg
import utils.utils_session as utilss

main_container = st.empty()
sidebar_container = st.sidebar

layout_choice = st.radio(
    "Choose layout:",
    ["Main Area", "Sidebar"],
    key="layout_choice"
)
if layout_choice == "Main Area":
    st.session_state.layout = main_container
else:
    st.session_state.layout = sidebar_container


with st.session_state.layout:
    st.info('Hello')
    st.info(st.session_state)
