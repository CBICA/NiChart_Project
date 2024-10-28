import os

import pandas as pd
import streamlit as st

# Initiate Session State Values
if "instantiated" not in st.session_state:

    # App type ('DESKTOP' or 'CLOUD')
    st.session_state.app_type = "CLOUD"
    st.session_state.app_type = "DESKTOP"

    # Dataframe to keep plot ids
    st.session_state.plots = pd.DataFrame(
        columns=["pid", "xvar", "yvar", "hvar", "trend", "centtype"]
    )
    st.session_state.plot_index = 1
    st.session_state.plot_active = ""

    # Study name
    st.session_state.dset = ""

    # Predefined paths for different tasks in the final results
    # The path structure allows nested folders with two levels
    # This should be good enough to keep results organized
    st.session_state.dict_paths = {
        "Lists": ["", "Lists"],
        "Dicoms": ["", "Dicoms"],
        "Nifti": ["", "Nifti"],
        "T1": ["Nifti", "T1"],
        "T2": ["Nifti", "T2"],
        "FL": ["Nifti", "FL"],
        "DTI": ["Nifti", "DTI"],
        "fMRI": ["Nifti", "fMRI"],
        "DLMUSE": ["", "DLMUSE"],
        "MLScores": ["", "MLScores"],
        "Plots": ["", "Plots"],
        "OutZipped": ["", "OutZipped"],
    }

    # Paths to input/output files/folders
    st.session_state.paths = {
        "root": "",
        "init": "",
        "last_in_dir": "",
        "target_dir": "",
        "target_file": "",
        "dset": "",
        "out": "",
        "Lists": "",
        "Nifti": "",
        "Dicoms": "",
        "user_dicoms": "",
        "T1": "",
        "user_T1": "",
        "T2": "",
        "FL": "",
        "DTI": "",
        "fMRI": "",
        "DLMUSE": "",
        "MLScores": "",
        "OutZipped": "",
        "Plots": "",
        "sel_img": "",
        "sel_seg": "",
        "csv_demog": "",
        "csv_dlmuse": "",
        "csv_plot": "",
        "csv_roidict": "",
        "csv_mlscores": "",
    }

    # Flags for various input/output
    st.session_state.flags = {
        "dset": False,
        "Dicoms": False,
        "dicom_series": False,
        "Nifti": False,
        "T1": False,
        "T2": False,
        "FL": False,
        "DTI": False,
        "fMRI": False,
        "DLMUSE": False,
        "csv_dlmuse": False,
        "csv_mlscores": False,
        "sel_img": False,
        "sel_mask": False
    }

    # Paths for output
    st.session_state.paths["root"] = os.path.dirname(os.path.dirname(os.getcwd()))
    st.session_state.paths["init"] = st.session_state.paths["root"]

    #########################################
    # FIXME : set to test folder outside repo
    st.session_state.paths["init"] = os.path.join(
        os.path.dirname(st.session_state.paths["root"]), "TestData"
    )

    st.session_state.paths["last_in_dir"] = st.session_state.paths["init"]

    # FIXME: This sets the default out path to a folder inside the root folder for now
    st.session_state.paths["out"] = os.path.join(
        st.session_state.paths["root"],
        "output_folder"
    )
    #########################################

    # Image modalities
    st.session_state.list_mods = ["T1", "T2", "FL", "DTI", "rMRI"]

    # Dictionaries
    tmp_dir = os.path.join(st.session_state.paths["root"], "resources", "MUSE")
    st.session_state.dicts = {
        "muse_derived": os.path.join(tmp_dir, "list_MUSE_mapping_derived.csv"),
        "muse_all": os.path.join(tmp_dir, "list_MUSE_all.csv"),
        # "muse_sel": os.path.join(tmp_dir, "list_MUSE_primary.csv"),
        "muse_sel": os.path.join(tmp_dir, "list_MUSE_all.csv"),
    }

    # Current roi dictionary
    st.session_state.roi_dict = None
    st.session_state.roi_dict_rev = None

    # Input image vars
    st.session_state.list_input_nifti = []

    # Dicom vars
    st.session_state.list_series = []
    st.session_state.num_dicom_scans = []
    st.session_state.df_dicoms = pd.DataFrame()
    st.session_state.sel_series = []
    st.session_state.sel_mod = "T1"

    # Default number of plots in a row
    st.session_state.max_plots_per_row = 5
    st.session_state.plots_per_row = 3

    # Image suffixes
    st.session_state.suff_t1img = "_T1.nii.gz"
    st.session_state.suff_seg = "_T1_DLMUSE.nii.gz"

    # Default values for plotting parameters
    st.session_state.plot_default_xvar = "Age"
    st.session_state.plot_default_yvar = "GM"
    st.session_state.plot_default_hvar = ""

    st.session_state.plot_xvar = ""
    st.session_state.plot_yvar = ""
    st.session_state.plot_hvar = ""

    st.session_state.trend_types = ["none", "ols", "lowess"]
    st.session_state.plot_trend = "none"

    st.session_state.cent_types = ["none", "CN-All", "CN-F", "CN-M"]
    st.session_state.plot_centtype = "none"

    # MRID selected by user
    st.session_state.sel_mrid = ""

    # Variable selected by user
    st.session_state.sel_var = ""

    # Debugging variables
    st.session_state.debug_show_state = False
    st.session_state.debug_show_paths = False
    st.session_state.debug_show_flags = False

    # Viewing variables


    st.session_state.instantiated = True

st.sidebar.image("../resources/nichart1.png")

st.write("# Welcome to NiChart Project!")

st.sidebar.info(
    """
                    Note: This website is based on materials from the [NiChart Project](https://neuroimagingchart.com/).
                    The content and the logo of NiChart are intellectual property of [CBICA](https://www.med.upenn.edu/cbica/).
                    Make sure that you read the [licence](https://github.com/CBICA/NiChart_Project/blob/main/LICENSE).
                    """
)

with st.sidebar.expander("Acknowledgments"):
    st.markdown(
        """
                The CBICA Dev team
                """
    )

st.sidebar.success("Select a task above")

with st.sidebar.expander('Flags'):

    if st.checkbox("Show paths?", value=False):
        st.session_state.debug_show_paths = True
    else:
        st.session_state.debug_show_paths = False

    if st.checkbox("Show flags?", value=False):
        st.session_state.debug_show_flags = True
    else:
        st.session_state.debug_show_flags = False

    if st.checkbox("Show all session state vars?", value=False):
        st.session_state.debug_show_state = True
    else:
        st.session_state.debug_show_state = False

    if st.checkbox("Switch to CLOUD?"):
        st.session_state.app_type = 'CLOUD'
    else:
        st.session_state.app_type = 'DESKTOP'

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

if st.session_state.debug_show_state:
    with st.expander("DEBUG: Session state - all variables"):
        st.write(st.session_state)

if st.session_state.debug_show_paths:
    with st.expander("DEBUG: Session state - paths"):
        st.write(st.session_state.paths)

if st.session_state.debug_show_flags:
    with st.expander("DEBUG: Session state - flags"):
        st.write(st.session_state.flags)
