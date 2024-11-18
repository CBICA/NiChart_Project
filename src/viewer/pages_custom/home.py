import os

import pandas as pd
import streamlit as st
import utils.utils_rois as utilroi
import utils.utils_st as utilst

#from wfork_streamlit_profiler import Profiler
# with Profiler():

# Initiate Session State Values
if "instantiated" not in st.session_state:

    ###################################
    # App type ('desktop' or 'cloud')
    st.session_state.app_type = "cloud"
    st.session_state.app_type = "desktop"
    st.session_state.app_config = {
        'cloud': {
            'msg_infile': 'Upload'
        },
        'desktop': {
            'msg_infile': 'Select'
        }
    }
    ###################################

    ###################################
    # General
    # Study name
    st.session_state.dset = ""

    # Icons for panels
    st.session_state.icon_thumb = {
        False: ':material/thumb_down:',
        True: ':material/thumb_up:'
    }

    # Flags for various i/o
    st.session_state.flags = {
        'dset': False,
        'dir_out': False,
        'dir_dicom': False,
        'dicom_series': False,
        'dir_nifti': False,
        'dir_t1': False,
        'dir_dlmuse': False,
        'csv_dlmuse': False,
        'csv_dlwmls': False,
        'csv_demog': False,
        'csv_dlmuse+demog': False,
        'dir_download': False,
        'csv_mlscores': False,
        'csv_plot': False,
    }

    # Predefined paths for different tasks in the final results
    # The path structure allows nested folders with two levels
    # This should be good enough to keep results organized
    st.session_state.dict_paths = {
        "lists": ["", "Lists"],
        "dicom": ["", "Dicoms"],
        "nifti": ["", "Nifti"],
        "T1": ["Nifti", "T1"],
        "T2": ["Nifti", "T2"],
        "FL": ["Nifti", "FL"],
        "DTI": ["Nifti", "DTI"],
        "fMRI": ["Nifti", "fMRI"],
        "dlmuse": ["", "DLMUSE"],
        "dlwmls": ["", "DLWMLS"],
        "mlscores": ["", "MLScores"],
        "plots": ["", "Plots"],
        "download": ["", "Download"],
    }

    # Paths to input/output files/folders
    st.session_state.paths = {
        "root": "",
        "init": "",
        "last_in_dir": "",
        "target_dir": "",
        "target_file": "",
        "dset": "",
        "dir_out": "",
        "lists": "",
        "dicom": "",
        "nifti": "",
        "T1": "",
        "T2": "",
        "FL": "",
        "DTI": "",
        "fMRI": "",
        "dlmuse": "",
        "dlwmls": "",
        "mlscores": "",
        "download": "",
        "plots": "",
        "sel_img": "",
        "sel_seg": "",
        "csv_demog": "",
        "csv_dlmuse": "",
        "csv_dlwmls": "",
        "csv_plot": "",
        "csv_roidict": "",
        "csv_mlscores": "",
    }

    # Flags to keep updates in user input/output
    st.session_state.is_updated = {
        "csv_plot": False,
    }

    # Set initial values for paths
    st.session_state.paths["root"] = os.path.dirname(os.path.dirname(os.getcwd()))
    st.session_state.paths["init"] = st.session_state.paths["root"]
    st.session_state.paths["dir_out"] = os.path.join(
        st.session_state.paths["root"],
        "output_folder"
    )
    if not os.path.exists(st.session_state.paths["dir_out"]):
        os.makedirs(st.session_state.paths["dir_out"])

    ############
    # FIXME : set init folder to test folder outside repo
    st.session_state.paths["init"] = os.path.join(
        os.path.dirname(st.session_state.paths["root"]), "TestData"
    )
    st.session_state.paths["last_in_dir"] = st.session_state.paths["init"]
    ############

    # Image modalities
    st.session_state.list_mods = ["T1", "T2", "FL", "DTI", "fMRI"]

    # Dictionaries
    res_dir = os.path.join(st.session_state.paths["root"], "resources")
    st.session_state.dicts = {
        "muse_derived": os.path.join(res_dir, "MUSE", "list_MUSE_mapping_derived.csv"),
        "muse_all": os.path.join(res_dir, "MUSE", "list_MUSE_all.csv"),
        # "muse_sel": os.path.join(res_dir, "MUSE", "list_MUSE_primary.csv"),
        "muse_sel": os.path.join(res_dir, "MUSE", "list_MUSE_all.csv"),
    }
    st.session_state.dict_categories = os.path.join(
        res_dir,
        'dictionaries',
        'var_categories',
        'dict_var_categories.json'
    )
    ###################################

    ###################################
    # Plotting
    # Dictionary with plot info
    st.session_state.plots = pd.DataFrame(
        columns=[
            "pid", "plot_type", "xvar", "yvar",
            "hvar", "hvals", "trend",
            "lowess_s", "traces", "centtype"
        ]
    )
    st.session_state.plot_index = 1
    st.session_state.plot_active = ""

    # Constant plot settings
    st.session_state.plot_const = {
        'trend_types' : ['', 'Linear', 'Smooth LOWESS Curve'],
        'centile_types' : ['', 'CN-All', 'CN-M', 'CN-F'],
        'linfit_trace_types' : ['data', 'lin_fit', 'conf_95%'],
        'distplot_trace_types' : ['histogram', 'density', 'rug'],
        'min_per_row': 1,
        'max_per_row': 5,
        'num_per_row': 3,
        'margin': 20,
        'h_init': 500,
        'h_coeff': 1.0,
        'h_coeff_max': 2.0,
        'h_coeff_min': 0.6,
        'h_coeff_step': 0.2,
        'distplot_binnum': 100
    }

    # Plot variables
    st.session_state.plot_var = {
        'df_data': pd.DataFrame(),
        'hide_settings': False,
        'hide_legend': False,
        'show_img': False,
        'plot_type': 'Scatter Plot',
        'xvar': '',
        'yvar': '',
        'hvar': '',
        'hvals': [],
        'trend': 'Linear',
        'traces': ['data', 'lin'],
        'lowess_s': 0.5,
        'centtype' : '',
    }
    ###################################

    ###################################
    # MRI view
    st.session_state.mriview_const = {
        'img_views': ["axial", "coronal", "sagittal"]
    }

    st.session_state.mriview_var = {
        'crop_to_mask': True,
        'show_overlay': True,
        'list_orient': ["axial", "coronal", "sagittal"]
    }

    ###################################

    ###################################
    # ROI dictionaries
    # List of roi names, indices, etc.
    st.session_state.rois = {
        'path': os.path.join(st.session_state.paths['root'], 'resources', 'lists'),
        'roi_dict_options': ['', 'muse_rois'], # This will be extended with additional roi dict.s
        'roi_csvs': {
            'muse_rois': 'MUSE_listROIs.csv',
            'muse_derived': 'MUSE_mapping_derivedROIs.csv'
        },
        'sel_roi_dict': 'muse_rois',
        'sel_derived_dict': 'muse_derived'
    }

    # Read initial roi lists (default:MUSE) to dictionaries
    ssroi = st.session_state.rois
    df_tmp = pd.read_csv(
        os.path.join(ssroi['path'], ssroi['roi_csvs'][ssroi['sel_roi_dict']])
    )
    dict1 = dict(zip(df_tmp["Index"].astype(str), df_tmp["Name"].astype(str)))
    dict2 = dict(zip(df_tmp["Name"].astype(str), df_tmp["Index"].astype(str)))
    dict3 = utilroi.muse_derived_to_dict(
        os.path.join(ssroi['path'], ssroi['roi_csvs'][ssroi['sel_derived_dict']])
    )
    st.session_state.rois['roi_dict'] = dict1
    st.session_state.rois['roi_dict_inv'] = dict2
    st.session_state.rois['roi_dict_derived'] = dict3
    ###################################

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

    # Image suffixes
    st.session_state.suff_t1img = "_T1.nii.gz"
    st.session_state.suff_flimg = "_FL.nii.gz"
    st.session_state.suff_seg = "_T1_DLMUSE.nii.gz"
    st.session_state.suff_dlwmls = "_FL_WMLS.nii.gz"

    # MRID selected by user
    st.session_state.sel_mrid = ""
    st.session_state.sel_roi = ""

    # Variable selected by user
    st.session_state.sel_var = ""

    # Variables selected by userroi_dict
    st.session_state.plot_sel_vars = []

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

# FIXME: For DEBUG
utilst.add_debug_panel()
