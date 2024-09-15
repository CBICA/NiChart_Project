import streamlit as st
from PIL import Image
import pandas as pd
import os

# Initiate Session State Values
if 'instantiated' not in st.session_state:

    # Dataframe to keep plot ids
    st.session_state.plots = pd.DataFrame({'PID':[]})
    st.session_state.pid = 1

    # Path to root folder
    st.session_state.dir_root = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))

    # Path to folder used as the init folder in file selection
    st.session_state.init_dir = st.session_state.dir_root

    # Default values for plotting parameters
    st.session_state.default_x_var = 'Age'
    st.session_state.default_y_var = 'GM'
    st.session_state.default_hue_var = 'Sex'
    st.session_state.trend_types = ['none', 'ols', 'lowess']
    st.session_state.default_trend_type = 'none'

    # ID selected by user
    st.session_state.sel_id = ''

    # Input fields for w_sMRI
    st.session_state.study_name = 'MyStudy'
    st.session_state.in_csv_MUSE = ''
    st.session_state.in_csv_Demog = ''

    # # FIXME: set default values for easy run
    # st.session_state.in_csv_MUSE = f'{st.session_state.dir_root}/test/test_input/test2_rois/Study1/Study1_DLMUSE.csv'
    # st.session_state.in_csv_Demog = f'{st.session_state.dir_root}/test/test_input/test2_rois/Study1/Study1_Demog.csv'

    # FIXME: Set these variables when the images are loaded or computed
    st.session_state.dir_t1img = st.session_state.dir_root + '/test/test_input/test3_nifti+roi'
    st.session_state.dir_dlmuse = st.session_state.dir_root + '/test/test_input/test3_nifti+roi'
    st.session_state.suffix_t1img = '_T1.nii.gz'
    st.session_state.suffix_dlmuse = '_T1_DLMUSE.nii.gz'

    # Path to out folder
    st.session_state.out_dir = ''

    # Input fields for plotting
    st.session_state.in_csv_sMRI = ''

    st.session_state.instantiated = True

st.sidebar.image("../resources/nichart1.png")

st.write("# Welcome to NiChart Project!")

st.sidebar.info("""
                    Note: This website is based on materials from the [NiChart Project](https://neuroimagingchart.com/).
                    The content and the logo of NiChart are intellectual property of [CBICA](https://www.med.upenn.edu/cbica/).
                    Make sure that you read the [licence](https://github.com/CBICA/NiChart_Project/blob/main/LICENSE).
                    """)

with st.sidebar.expander("Acknowledgments"):
    st.markdown("""
                The CBICA Dev team
                """)

st.sidebar.success("Select a task above")

st.markdown(
        """
    NiChart is an open-source framework built specifically for
    deriving Machine Learning based indices from MRI data.

    **👈 Select a task from the sidebar** to process, analyze and visualize your data!

    ### Want to learn more?
    - Check out [NiChart Web page](https://neuroimagingchart.com)
    - Visit [NiChart GitHub](https://github.com/CBICA/NiChart_Project)
    - Jump into our [documentation](https://github.com/CBICA/NiChart_Project)
    - Ask a question in our [community
        forums](https://github.com/CBICA/NiChart_Project)
        """
)

st.markdown(
            """
            You can try NiChart manually via our github
            ```bash
            git clone https://github.com/CBICA/NiChart_Project
            git submodule update --init --recursive --remote
            pip install -r requirements.txt
            ```

            And to run the workflows, just run:
            ```bash
            python3 run.py --dir_input input folder --dir_output output_folder --studies 1 --version my_version --cores 4 --conda 0
            ```

            You can always find more options at our documentation
            """
)
