import streamlit as st
from PIL import Image
import pandas as pd
import os

# Initiate Session State Values
if 'instantiated' not in st.session_state:

    # Dataframe to keep plot ids
    st.session_state.plots = pd.DataFrame(columns = ['pid', 'xvar', 'yvar', 'hvar',
                                                     'trend', 'centtype'])
    st.session_state.plot_index = 1
    st.session_state.plot_active = ''

    # Study name
    st.session_state.dset_name = ''

    # Path to root folder
    st.session_state.path_root = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))

    # Path to init folder
    st.session_state.path_init = os.path.join(st.session_state.path_root, 'test')

    # Path to last user selection (to initialize file/folder selector)
    st.session_state.path_last_sel = st.session_state.path_init

    st.session_state.path_sel_img = ''
    st.session_state.path_sel_mask = ''

    # Paths to output files/folders
    st.session_state.path_out = ''
    st.session_state.path_dset = ''
    st.session_state.path_nifti = ''
    st.session_state.path_selmod = ''
    st.session_state.path_t1 = ''
    st.session_state.path_t2 = ''
    st.session_state.path_fl = ''
    st.session_state.path_dti = ''
    st.session_state.path_fmri = ''
    st.session_state.path_dlmuse = ''
    st.session_state.path_mlscore = ''
    st.session_state.path_csv_demog = ''
    st.session_state.path_csv_dlmuse = ''
    st.session_state.path_csv_mlscores = ''
    st.session_state.path_csv_viewdlmuse = ''

    # Input image vars
    st.session_state.list_input_nifti = []

    
    # Dicom vars
    st.session_state.path_dicom = ''
    st.session_state.list_series = []    
    st.session_state.df_dicoms = pd.DataFrame()
    st.session_state.sel_series = []
    st.session_state.sel_mod = ''

    #####
    # FIXME : for quick test
    #st.session_state.path_csv_mlscores = st.session_state.path_root + '/test/test3_nifti+roi/output/MyStudy/MLScores/MyStudy_DLMUSE+MLScores.csv'   
    st.session_state.path_last_sel = st.session_state.path_init
    st.session_state.path_last_sel = st.session_state.path_init + '/../../TestData'
    #####

    # Image suffixes
    st.session_state.suff_t1img = '_T1.nii.gz'
    st.session_state.suff_dlmuse = '_T1_DLMUSE.nii.gz'

    # Default values for plotting parameters
    st.session_state.plot_xvar = 'Age'
    st.session_state.plot_yvar = 'GM'
    st.session_state.plot_hvar = 'Sex'

    st.session_state.trend_types = ['none', 'ols', 'lowess']
    st.session_state.plot_trend = 'none'

    st.session_state.cent_types = ['none', 'CN-All', 'CN-F', 'CN-M']
    st.session_state.plot_centtype = 'none'

    # MRID selected by user
    st.session_state.sel_mrid = ''

    # Variable selected by user
    st.session_state.sel_var = ''

    # MUSE dictionaries
    st.session_state.dir_resources = os.path.join(st.session_state.path_root, 'resources')

    st.session_state.dict_muse_derived = os.path.join(st.session_state.dir_resources, 'MUSE',
                                                      'list_MUSE_mapping_derived.csv')
    st.session_state.dict_muse_all = os.path.join(st.session_state.dir_resources, 'MUSE',
                                                      'list_MUSE_all.csv')
    
    st.session_state.dict_muse_sel = os.path.join(st.session_state.dir_resources, 
                                                       'MUSE', 'list_MUSE_primary.csv')


    # ########################################################
    # ## FIXME : this is for quickly loading example test data
    # st.session_state.path_csv_spare = st.session_state.path_root + '/test/test4_adni3/output/out_combined/MyStudy_All.csv'
    # st.session_state.path_out = st.session_state.path_root + '/test/test4_adni3/output'

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
