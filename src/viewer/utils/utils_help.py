import os
import shutil
from typing import Any

import numpy as np
import streamlit as st
import utils.utils_io as utilio
import utils.utils_session as utilses

def util_help_dialog(
    s_title: str,
    s_text: str,
    s_warning = '': str
) -> None:
    @st.dialog(s_title)  # type:ignore
    def help_working_dir():
        st.markdown(s_text)
        if s_warning != '':
            st.warning(s_warning)

    col1, col2 = st.columns([0.5, 0.1])
    with col2:
        if st.button(
            "Get help 🤔", key="key_btn_help_" + s_title, use_container_width=True
        ):
            help_working_dir()


def util_help_button(s_title: str) -> None:
    col1, col2 = st.columns([0.5, 0.1])
    with col2:
        if st.button(
            "Get help 🤔",
            key="key_btn_help_" + s_title,
            use_container_width=True
        ):
            help_working_dir()

def util_help_workingdir() -> None:
    s_title="Working Directory"
    
    def help_working_dir():
        st.markdown(
            """
            - A NiChart pipeline executes a series of steps, with input/output files organized in a predefined folder structure.

            - Results for an **experiment** (a new analysis on a new dataset) are kept in a dedicated **working directory**.

            - Set an **"output path"** (desktop app only) and an **"experiment name"** to define the **working directory** for your analysis. You only need to set the working directory once.

            - The **experiment name** can be any identifier that describes your analysis or data; it does not need to match the input study or data folder name.

            - You can initiate a NiChart pipeline by selecting the **working directory** from a previously completed experiment.
            """
        )
        st.warning(
            """
            On the cloud app, uploaded data and results of experiments are deleted in regular intervals!

            Accordingly, the data for a previous experiment may not be available.
            """
        )

    col1, col2 = st.columns([0.5, 0.1])
    with col2:
        if st.button(
            "Get help 🤔", key="key_btn_help_working_dir", use_container_width=True
        ):
            help_working_dir()


