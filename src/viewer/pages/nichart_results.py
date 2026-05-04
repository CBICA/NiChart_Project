import streamlit as st
import utils.utils_pages as utilpg
import utils.utils_session as utilses
import gui.utils_plots as utilpl
import gui.utils_dataprep as utildataprep
from utils.utils_styles import inject_global_css
from utils.utils_logger import setup_logger
import os

logger = setup_logger()
logger.debug('Page: Results')

inject_global_css()
utilpg.set_global_style()

# if 'instantiated' not in st.session_state or not st.session_state.instantiated:
#     utilses.init_session_state()

ss= st.session_state

with st.container(
    horizontal=True, horizontal_alignment="center", width='stretch'
):
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Overview", "Reference Charts", "My Data", "My Charts", "My Reports"],
        on_change='rerun',
    )

## Info
if tab1.open:
    with tab1:
        st.info(
            '''
            **View NiChart**, Neuroimaging Chart of Age-Related Brain Changes
            - **Reference Charts:** Explore normative brain aging centiles (**user data not required**)
            - **My Data:** Prepare and validate user datasets.
            - **My Charts:** View personalized age plots with reference centiles.
            - **My Reports:** Generate personalized reports.
            ''', icon=':material/info:'
        )

## Data selection
if tab2.open:
    with tab2:
        # Set viewer to user data
        ss.flag_user_data = False

        # Load centile data if not already loaded
        if ss.ref_plots['data']['df_cent'] is None:
            utildataprep.load_centile_data('ref_plots')

        # View data
        utilpl.panel_view_vars()

## Data selection
if tab3.open:
    with tab3:
        # Prepare user data
        utildataprep.panel_prepare_data()

## View charts
if tab4.open:
    with tab4:
        # Set viewer to user data
        ss.flag_user_data = True

        # View data
        utilpl.panel_view_vars()

## Data selection
if tab5.open:
    st.write('Reporting coming soon ...')

if ss.mode == 'debug':
    utilses.disp_session_state()
