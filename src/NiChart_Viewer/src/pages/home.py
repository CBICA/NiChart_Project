import os

import pandas as pd
import streamlit as st

# Initiate Session State Values
if "instantiated" not in st.session_state:

    # Dataframe to keep plot ids
    st.session_state.plots = pd.DataFrame(
        columns=["pid", "xvar", "yvar", "hvar", "trend", "centtype"]
    )
    st.session_state.plot_index = 1
    st.session_state.plot_active = ""

    # Study name
    st.session_state.dset_name = ""

    # Paths to input/output files/folders
    st.session_state.paths = {
        "root": "",
        "init": "",
        "last_sel": "",
        "dset": "",
        "out": "",
        "nifti": "",
        "dicom": "",
        "T1": "",
        "T2": "",
        "FL": "",
        "DTI": "",
        "fMRI": "",
        "dlmuse": "",
        "mlscores": "",
        "plots": "",
        "sel_img": "",
        "sel_mask": "",
        "csv_demog": "",
        "csv_dlmuse": "",
        "csv_plot": "",
        "csv_mlscores": "",
        "csv_viewdlmuse": "",
    }
    st.session_state.paths["root"] = os.path.dirname(
        os.path.dirname(os.path.dirname(os.getcwd()))
    )
    st.session_state.paths["init"] = os.path.join(
        st.session_state.paths["root"], "test"
    )
    st.session_state.paths["last_sel"] = st.session_state.paths["init"]

    #########################################
    # FIXME : for quick test
    # st.session_state.paths['csv_mlscores'] = st.session_state.paths['root'] + '/test/test3_nifti+roi/output/MyStudy/MLScores/MyStudy_DLMUSE+MLScores.csv'
    st.session_state.paths["last_sel"] = (
        st.session_state.paths["init"] + "/../../TestData"
    )
    #########################################

    # Image modalities
    st.session_state.list_mods = ["T1", "T2", "FL", "DTI", "rMRI"]

    # Dictionaries
    tmp_dir = os.path.join(st.session_state.paths["root"], "resources", "MUSE")
    st.session_state.dicts = {
        "muse_derived": os.path.join(tmp_dir, "list_MUSE_mapping_derived.csv"),
        "muse_all": os.path.join(tmp_dir, "list_MUSE_all.csv"),
        "muse_sel": os.path.join(tmp_dir, "list_MUSE_primary.csv"),
    }

    # Input image vars
    st.session_state.list_input_nifti = []

    # Dicom vars
    st.session_state.list_series = []
    st.session_state.df_dicoms = pd.DataFrame()
    st.session_state.sel_series = []
    st.session_state.sel_mod = "T1"

    # Default number of plots in a row
    st.session_state.max_plots_per_row = 5
    st.session_state.plots_per_row = 3

    # Image suffixes
    st.session_state.suff_t1img = "_T1.nii.gz"
    st.session_state.suff_dlmuse = "_T1_DLMUSE.nii.gz"

    # Default values for plotting parameters
    st.session_state.plot_xvar = "Age"
    st.session_state.plot_yvar = "GM"
    st.session_state.plot_hvar = "Sex"

    st.session_state.trend_types = ["none", "ols", "lowess"]
    st.session_state.plot_trend = "none"

    st.session_state.cent_types = ["none", "CN-All", "CN-F", "CN-M"]
    st.session_state.plot_centtype = "none"

    # MRID selected by user
    st.session_state.sel_mrid = ""

    # Variable selected by user
    st.session_state.sel_var = ""

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

with st.expander("FIXME: TMP - Session state"):
    st.write(st.session_state)
