import streamlit as st
import utils.utils_pages as utilpg
import utils.utils_session as utilses
import gui.utils_results as utilres
import gui.utils_navig as utilnav
import utils.utils_settings as utilset
from utils.utils_styles import inject_global_css
from utils.utils_logger import setup_logger
import os

logger = setup_logger()
logger.debug('Page: Results')

inject_global_css()
utilpg.set_global_style()

if 'instantiated' not in st.session_state or not st.session_state.instantiated:
    utilses.init_session_state()

with st.container(
    horizontal=True, horizontal_alignment="center", width='stretch'
):
    tab1, tab2, tab3 = st.tabs(
        ["Overview", "Datasets", "View Charts"],
        on_change='rerun',
    )

## Info
if tab1.open:
    with tab1:
        st.markdown("<h5 style='text-align:center; color:#3a3a88;'>Reference Charts\n\n</h1>", unsafe_allow_html=True)

        cols = st.columns([1,6,1])
        with cols[1]:
            st.markdown(
                '''
                The **Reference Charts** page shows age trends of reference values for brain imaging biomarkers.

                - **Reference group**: choose the population to compare against
                  (all cognitively normal, males only, females only, or ICV-normalized)
                
                - **Variable**: select the imaging measure or biomarker to display
                
                - **Centile bands**: the shaded regions show the 5th–95th and 25th–75th centile ranges across age; the solid line is the median (50th centile)
                ''', unsafe_allow_html=True
            )

## Data description
if tab2.open:
    with st.container(border=True, horizontal_alignment = 'left'):
        st.image(os.path.join(
            st.session_state.paths['resources'], 'images', 'nichart_data.png'
        ), width=1200
        )
## Charts
if tab3.open:
    with tab3:
        utilres.panel_centile_view()

if st.session_state.mode == 'debug':
    utilses.disp_session_state()
