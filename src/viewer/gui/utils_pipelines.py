import os
import shutil
import time
from typing import Any

import pandas as pd
import numpy as np
import nibabel as nib
from nibabel.orientations import axcodes2ornt, ornt_transform
from scipy import ndimage
import utils.utils_misc as utilmisc
import utils.utils_user_select as utiluser
import utils.utils_io as utilio

import utils.utils_session as utilses
import gui.utils_plots as utilpl
import gui.utils_mriview as utilmri
import gui.utils_view as utilview
import pandas as pd
import gui.utils_widgets as utilwd

import streamlit_antd_components as sac

import streamlit as st
from stqdm import stqdm

from utils.utils_logger import setup_logger
logger = setup_logger()

#################################
## Function definitions
def show_description(pipeline) -> None:
    """
    Panel for viewing pipeline description
    """
    with st.container(border=True, height=300):
        f_logo = os.path.join(
            st.session_state.paths['resources'], 'pipelines', pipeline, f'logo_{pipeline}.png'
        )
        fdoc = os.path.join(
            st.session_state.paths['resources'], 'pipelines', pipeline, f'overview_{pipeline}.md'
        )
        cols = st.columns([6, 1])
        with cols[0]:
            with open(fdoc, 'r') as f:
                st.markdown(f.read())
        with cols[1]:
            st.image(f_logo)

def select_pipeline():
    '''
    Select a pipeline and show overview
    '''
    st.markdown("##### Select:")

    sac.divider(key='_p2_div1')

    pipelines = st.session_state.pipelines
    pnames = pipelines.Name.tolist()

    sel_opt = sac.chip(
        pnames,
        label='', index=0, align='left',
        size='md', radius='md', multiple=False, color='cyan',
        description='Select a pipeline'
    )

    show_description(sel_opt.lower())


def pipeline_menu():
    cols = st.columns([10,1,10])
    out_dir = os.path.join(
        st.session_state.paths['out_dir'], st.session_state['prj_name']
    )

    with cols[0]:
        select_pipeline()
    # with cols[2]:

def panel_pipelines():

    workflow = st.session_state.workflow

    if workflow is None:
        st.info('Please select a Workflow!')
        return

    with st.container(horizontal=True, horizontal_alignment="center"):
        st.markdown("<h4 style=color:#3a3a88;'>Select and Run Pipeline\n\n</h1>", unsafe_allow_html=True, width='content')

    if st.session_state.workflow == 'ref_data':
        st.info('''
            You’ve selected the **Reference Data** workflow. This option doesn’t require pipeline selection.
            - If you meant to analyze your data, please go back and choose a different workflow.
            - Otherwise, continue to the next step to explore the reference values.
            '''
        )
    else:
        pipeline_menu()

