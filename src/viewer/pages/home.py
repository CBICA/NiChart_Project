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

#st.info(
st.markdown(
    """
    ### Welcome to NiChart Project!
    - NiChart is an **<u>open-source framework</u>** built specifically for deriving **<u>machine learning biomarkers </u>** from **<u>MRI imaging data</u>**.
    """
    , unsafe_allow_html=True
)

st.image("../resources/nichart1.png", width=300)

sel = st.pills(
    'Select an option',
    ['Overview', 'Quick Start Guide', 'Links', 'Installation'],
    default = 'Overview',
    label_visibility="collapsed"
)

if sel == 'Overview':
    with st.container(border=True):
        st.markdown(
            """
            - NiChart is a modular platform offering tools for **<u>image processing </u>** and **<u>data analysis</u>**.

            - Users can extract **<u>imaging phenotypes </u>** and **<u>machine learning (ML) indices </u>** of disease and aging.

            - Pre-trained **<u>ML models </u>** allow users to quantify complex brain changes and compare results against **<u>normative and disease-specific reference ranges</u>**.
            """
            , unsafe_allow_html=True
        )

elif sel == 'Quick Start Guide':
    with st.container(border=True):
        st.markdown(
            """
            ##### Explore Brain Chart (No Data Upload Required):
            
            - **<u>Explore the distribution</u>** of imaging variables and machine learning–derived biomarkers **<u>from the NiChart reference dataset</u>**.
            
            - This module is designed for visualization only and **<u>does not require user data**</u>.
            
            - **<u>Includes:</u>** brain segmentation, region volumes, and biomarkers for aging and disease (e.g., AD, brain age).
            
            ##### Analyze Your Own Data:

            - **<u>Upload Your Data: </u>** Navigate to the "Data" page to upload the files you wish to analyze.

            - **<u>Select Your Pipeline: </u>** Go to the "Pipelines" page and choose the analysis workflow you want to apply to your data.

            - **<u>View and/or Download Your Results: </u>** Once the pipeline has finished processing, your findings will be available to view or to download.


            """
            , unsafe_allow_html=True
        )

elif sel == 'Links':
    with st.container(border=True):
        st.markdown(
            """
            - Check out [NiChart Web page](https://neuroimagingchart.com)
            - Visit [NiChart GitHub](https://github.com/CBICA/NiChart_Project)
            - Jump into [our documentation](https://cbica.github.io/NiChart_Project)
            - Ask a question in [our community discussions](https://github.com/CBICA/NiChart_Project/discussions)
            """
            , unsafe_allow_html=True
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
            , unsafe_allow_html=True
        )
    
