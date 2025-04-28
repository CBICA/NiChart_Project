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
    - NiChart is an ***:red[open-source framework]*** built specifically for deriving ***:red[machine learning biomarkers]*** from ***:red[MRI imaging data]***.
    """
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
            - NiChart is a modular platform offering tools for ***:red[image processing]*** and ***:red[data analysis]***.

            - Users can extract ***:red[imaging phenotypes]*** and ***:red[machine learning (ML) indices]*** of disease and aging.

            - Pre-trained ***:red[ML models]*** allow users to quantify complex brain changes and compare results against ***:red[normative and disease-specific reference ranges]***.
            """
        )

elif sel == 'Quick Start Guide':
    with st.container(border=True):
        st.markdown(
            """
            ##### Analyze Your Own Data:

            - ***:red[Quick Setup:]*** Select a few settings that will help to better organize your work.

            - ***:red[Upload Your Data:]*** Navigate to the "Data" page to upload the files you wish to analyze.

            - ***:red[Select Your Pipeline:]*** Go to the "Pipelines" page and choose the analysis workflow you want to apply to your data.

            - ***:red[View Your Results:]*** Once the pipeline has finished processing, your findings will be available on the "Results" page.

            ##### Explore Sample Outputs (No Data Upload Required):

            - ***:red[View Examples:]*** Go to the "Examples" page to visualize results of image processing and ML steps on public samples

            - ***:red[Contents:]*** Segmentation of brain anatomy, volumes of brain regions, biomarkers of disease and aging such as AD and Brain Aging indices.
            """
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
    
