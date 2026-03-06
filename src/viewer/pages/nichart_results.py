import streamlit as st
import utils.utils_pages as utilpg
import utils.utils_session as utilses
import gui.utils_results as utilres
import gui.utils_navig as utilnav
import utils.utils_settings as utilset
from utils.utils_styles import inject_global_css
from utils.utils_logger import setup_logger

logger = setup_logger()
logger.debug('Page: Results')

inject_global_css()
utilpg.set_global_style()

if 'instantiated' not in st.session_state or not st.session_state.instantiated:
    utilses.init_session_state()

@st.dialog("Help Information", width="medium")
def my_help():
    st.write("""
    The **Results** page shows age trends of reference values for brain imaging biomarkers.

    - **Reference group**: choose the population to compare against
      (all cognitively normal, males only, females only, or ICV-normalized)

    - **Variable**: select the imaging measure or biomarker to display

    - **Centile bands**: the shaded regions show the 5th–95th and 25th–75th centile
      ranges across age; the solid line is the median (50th centile)
    """)

with st.container(horizontal=True, horizontal_alignment="center"):
    st.markdown("<h4 style='color:#3a3a88;'>Reference Centiles</h4>", unsafe_allow_html=True, width='content')

utilres.panel_centile_view()

if st.session_state.workflow == 'ref_data':
    utilnav.main_navig(
        'Info', 'pages/nichart_ref_data.py',
        'Home', 'pages/nichart_home.py',
        utilset.edit_settings, my_help
    )
else:
    utilnav.main_navig(
        'Pipelines', 'pages/nichart_pipelines.py',
        'Home', 'pages/nichart_home.py',
        utilset.edit_settings, my_help
    )

if st.session_state.mode == 'debug':
    utilses.disp_session_state()
