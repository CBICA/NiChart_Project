import streamlit as st
import utils.utils_pages as utilpg
import utils.utils_session as utilss
import logging

from utils.utils_logger import setup_logger

# Page config should be called for each page
utilpg.config_page()
utilpg.show_menu()
utilpg.add_sidebar_options()

logger = setup_logger()

logger.debug('Start of Home Screen!')


# utilpg.select_main_menu()

st.info(
    """
    ### Welcome to NiChart Project!
    - NiChart is an :red[open-source framework] built specifically for deriving :red[machine learning biomarkers] from MRI imaging data.
    """
)

st.image("../resources/nichart1.png", width=300)

sel = st.pills(
    'Select an option',
    ['Links', 'Installation'],
    default = 'Links',
    label_visibility="collapsed"
)

if sel == 'Links':
    with st.container(border=True):
        st.markdown(
            """
            - Check out [NiChart Web page](https://neuroimagingchart.com)
            - Visit [NiChart GitHub](https://github.com/CBICA/NiChart_Project)
            - Jump into [our documentation](https://cbica.github.io/NiChart_Project)
            - Ask a question in [our community discussions](https://github.com/CBICA/NiChart_Project/discussions)
            """
        )

elif sel == 'Installation':
    with st.container(border=True):
    #with st.expander(label='Installation'):
        st.markdown(
            """
            - You can install NiChart Project desktop
            ```
            pip install NiChart_Project
            ```

            - Run the application
            ```
            cd src/viewer
            streamlit run NiChartProject.py
            ```

            - Alternatively, the cloud app can be launched at
            ```
            https://cloud.neuroimagingchart.com
            ```
            """
        )
    
