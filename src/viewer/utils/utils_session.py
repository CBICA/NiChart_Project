import argparse
import os
import pandas as pd
import streamlit as st
import utils.utils_rois as utilroi
import utils.utils_st as utilst
import plotly.express as px

def config_page():
    st.set_page_config(
        page_title="NiChart",
        page_icon=st.session_state.nicon,
        layout="wide",
        #layout="centered",
        menu_items={
            "Get help": "https://neuroimagingchart.com/",
            "Report a bug": "https://neuroimagingchart.com/",
            "About": "https://neuroimagingchart.com/",
        },
    )

def init_session_state() -> None:
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
        # Pipelines
        st.session_state.pipelines = [
            'Home',
            'sMRI Biomarkers (T1)',
            'WM Lesion Segmentation (FL)',
            'DTI Biomarkers (DTI)',
            'Resting State fMRI Biomarkers (rsfMRI)',
        ]
        st.session_state.pipeline = 'Home'
        st.session_state._pipeline = st.session_state.pipeline
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
            "file_search_dir": "",
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
        st.session_state.paths["file_search_dir"] = st.session_state.paths["init"]
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
            'lists',
            'dict_var_categories.json'
        )
        
        # Average ICV estimated from a large sample
        # IMPORTANT: Used in NiChart Engine for normalization!
        st.session_state.mean_icv = 1430000      
        ###################################

        ###################################
        # Plotting
        # Dictionary with plot info
        st.session_state.plots = pd.DataFrame(
            columns=[
                "pid", "plot_type", "xvar", "yvar",
                "hvar", "hvals", "corr_icv", "plot_centiles", "trend",
                "lowess_s", "traces", "centtype"
            ]
        )
        st.session_state.plot_index = 1
        st.session_state.plot_active = ""

        # Constant plot settings
        st.session_state.plot_const = {
            'trend_types' : ['', 'Linear', 'Smooth LOWESS Curve'],
            'centile_types' : ['', 'CN', 'CN_Males', 'CN_Females', 'CN_ICV_Corrected'],
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
            'corr_icv': False,
            'plot_centiles': False,
            'trend': 'Linear',
            'traces': ['data', 'lin'],
            'lowess_s': 0.5,
            'centtype' : '',
            'h_coeff': 1.0        
        }
        ###################################

        ###################################
        # Color maps for plots
        st.session_state.plot_colors = {
            'data': px.colors.qualitative.Set1,
            'centile': [
                "rgba(0, 0, 120, 0.5)",
                "rgba(0, 0, 90, 0.7)",
                "rgba(0, 0, 60, 0.9)",
                "rgba(0, 0, 90, 0.7)",
                "rgba(0, 0, 120, 0.5)"
            ]
        }
        ###################################

        ###################################
        # MRI view
        st.session_state.mriview_const = {
            'img_views': ["axial", "coronal", "sagittal"],
            'w_init': 500,
            'w_coeff': 1.0,
            'w_coeff_max': 2.0,
            'w_coeff_min': 0.6,
            'w_coeff_step': 0.2
        }

        st.session_state.mriview_var = {
            'crop_to_mask': True,
            'show_overlay': True,
            'list_orient': ["axial", "coronal", "sagittal"],
            'w_coeff': 1.0        
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

def update_default_paths() -> None:
    """
    Update default paths in session state if the working dir changed
    """
    for d_tmp in st.session_state.dict_paths.keys():
        st.session_state.paths[d_tmp] = os.path.join(
            st.session_state.paths["dset"],
            st.session_state.dict_paths[d_tmp][0],
            st.session_state.dict_paths[d_tmp][1],
        )
        print(f"setting {st.session_state.paths[d_tmp]}")

    st.session_state.paths["csv_dlmuse"] = os.path.join(
        st.session_state.paths["dlmuse"], "DLMUSE_Volumes.csv"
    )

    st.session_state.paths["csv_mlscores"] = os.path.join(
        st.session_state.paths["mlscores"],
        f"{st.session_state.dset}_DLMUSE+MLScores.csv",
    )

    st.session_state.paths["csv_demog"] = os.path.join(
        st.session_state.paths["dset"], "lists", "Demog.csv"
    )

    st.session_state.paths["csv_plot"] = os.path.join(
        st.session_state.paths["plots"], "Data.csv"
    )


def reset_flags() -> None:
    """
    Resets flags if the working dir changed
    """
    for tmp_key in st.session_state.flags.keys():
        st.session_state.flags[tmp_key] = False
    st.session_state.flags["dset"] = True
