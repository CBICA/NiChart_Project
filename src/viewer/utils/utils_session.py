import os
import shutil
from typing import Any

import jwt
import pandas as pd
import plotly.express as px
import streamlit as st
import utils.utils_io as utilio
import utils.utils_rois as utilroi
from PIL import Image

# from streamlit.web.server.websocket_headers import _get_websocket_headers


def config_page() -> None:
    st.session_state.nicon = Image.open("../resources/nichart1.png")
    st.set_page_config(
        page_title="NiChart",
        page_icon=st.session_state.nicon,
        layout="wide",
        # layout="centered",
        menu_items={
            "Get help": "https://neuroimagingchart.com/",
            "Report a bug": "https://github.com/CBICA/NiChart_Project/issues/new?assignees=&labels=&projects=&template=bug_report.md&title=%5BBUG%5D+",
            "About": "https://neuroimagingchart.com/",
        },
    )


# Function to parse AWS login (if available)
def process_session_token() -> Any:
    # headers = _get_websocket_headers()
    headers = st.context.headers
    if not headers or "X-Amzn-Oidc-Data" not in headers:
        return {}
    return jwt.decode(
        headers["X-Amzn-Oidc-Data"],
        algorithms=["ES256"],
        options={"verify_signature": False},
    )


def process_session_user_id() -> Any:
    headers = st.context.headers
    if not headers or "X-Amzn-Oidc-Identity" not in headers:
        return "NO_USER_FOUND"
    return headers["X-Amzn-Oidc-Identity"]


def init_session_state() -> None:
    # Initiate Session State Values
    if "instantiated" not in st.session_state:

        ###################################
        # App type ('desktop' or 'cloud')
        if os.getenv("NICHART_FORCE_CLOUD", "0") == "1":
            st.session_state.forced_cloud = True
            st.session_state.app_type = "cloud"
        else:
            st.session_state.forced_cloud = False
            st.session_state.app_type = "desktop"

        st.session_state.app_config = {
            "cloud": {"msg_infile": "Upload"},
            "desktop": {"msg_infile": "Select"},
        }

        st.session_state.nicon = Image.open("../resources/nichart1.png")

        st.session_state.sel_main_menu = "Home"
        st.session_state.sel_pipeline = None
        st.session_state.sel_pipeline_step = None

        # Store user session info for later retrieval
        if st.session_state.app_type == "cloud":
            st.session_state.cloud_session_token = process_session_token()
            if st.session_state.cloud_session_token:
                st.session_state.has_cloud_session = True
                st.session_state.cloud_user_id = process_session_user_id()
            else:
                st.session_state.has_cloud_session = False
        else:
            st.session_state.has_cloud_session = False

        ###################################

        ###################################
        # Pipelines
        st.session_state.pipelines = [
            "Home",
            "sMRI Biomarkers (T1)",
            "WM Lesion Segmentation (FL)",
            "DTI Biomarkers (DTI)",
            "Resting State fMRI Biomarkers (rsfMRI)",
        ]
        st.session_state.pipeline = "Home"
        st.session_state._pipeline = st.session_state.pipeline
        ###################################

        ###################################
        # General
        # Study name
        st.session_state.experiment = ""

        # Icons for panels
        st.session_state.icon_thumb = {
            False: ":material/thumb_down:",
            True: ":material/thumb_up:",
        }

        # Flags for various i/o
        st.session_state.flags = {
            "experiment": False,
            "dicoms": False,
            "dicoms_series": False,
            "nifti": False,
            "T1": False,
            "dlmuse": False,
            "dlmuse_csv": False,
            "dlwmls_csv": False,
            "demog_csv": False,
            "dlmuse+demog_csv": False,
            "downloaddir": False,
            "mlscores_csv": False,
            "plot_csv": False,
        }

        # Predefined paths for different tasks in the final results
        # The path structure allows nested folders with two levels
        # This should be good enough to keep results organized
        st.session_state.dict_paths = {
            "lists": ["", "Lists"],
            "dicoms": ["", "Dicoms"],
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
            "dir_out": "",
            "experiment": "",
            "lists": "",
            "dicoms": "",
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
            "demog_csv": "",
            "dlmuse_csv": "",
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
        if st.session_state.has_cloud_session:
            user_id = st.session_state.cloud_user_id
            st.session_state.paths["dir_out"] = os.path.join(
                st.session_state.paths["root"], "output_folder", user_id
            )
        else:
            st.session_state.paths["dir_out"] = os.path.join(
                st.session_state.paths["root"], "output_folder"
            )

        if not os.path.exists(st.session_state.paths["dir_out"]):
            os.makedirs(st.session_state.paths["dir_out"])

        # Copy demo folders into user folders as needed
        if st.session_state.has_cloud_session:
            # Copy demo dirs to user folder (TODO: make this less hardcoded)
            demo_dir_paths = [
                os.path.join(
                    st.session_state.paths["root"],
                    "output_folder",
                    "NiChart_sMRI_Demo1",
                ),
                os.path.join(
                    st.session_state.paths["root"],
                    "output_folder",
                    "NiChart_sMRI_Demo2",
                ),
            ]
            for demo in demo_dir_paths:
                demo_name = os.path.basename(demo)
                destination_path = os.path.join(
                    st.session_state.paths["dir_out"], demo_name
                )
                if os.path.exists(destination_path):
                    shutil.rmtree(destination_path)
                shutil.copytree(demo, destination_path, dirs_exist_ok=True)

        ############
        # FIXME : set init folder to test folder outside repo
        st.session_state.paths["init"] = os.path.join(
            st.session_state.paths["root"], "test_data"
        )
        st.session_state.paths["file_search_dir"] = st.session_state.paths["init"]
        ############

        # Image modalities
        st.session_state.list_mods = ["T1", "T2", "FL", "DTI", "fMRI"]

        # Dictionaries
        res_dir = os.path.join(st.session_state.paths["root"], "resources")
        st.session_state.dicts = {
            "muse_derived": os.path.join(
                res_dir, "MUSE", "list_MUSE_mapping_derived.csv"
            ),
            "muse_all": os.path.join(res_dir, "MUSE", "list_MUSE_all.csv"),
            # "muse_sel": os.path.join(res_dir, "MUSE", "list_MUSE_primary.csv"),
            "muse_sel": os.path.join(res_dir, "MUSE", "list_MUSE_all.csv"),
        }
        st.session_state.dict_categories = os.path.join(
            res_dir, "lists", "dict_var_categories.json"
        )

        # Various parameters

        # Average ICV estimated from a large sample
        # IMPORTANT: Used in NiChart Engine for normalization!
        st.session_state.mean_icv = 1430000

        # Min number of samples to run harmonization
        st.session_state.harm_min_samples = 30

        ###################################

        ###################################
        # Plotting
        # Dictionary with plot info
        st.session_state.plots = pd.DataFrame(
            columns=[
                "pid",
                "plot_type",
                "xvar",
                "xmin",
                "xmax",
                "yvar",
                "ymin",
                "ymax",
                "hvar",
                "hvals",
                "corr_icv",
                "plot_cent_normalized",
                "trend",
                "lowess_s",
                "traces",
                "centtype",
            ]
        )
        st.session_state.plot_index = 1
        st.session_state.plot_active = ""

        # Constant plot settings
        st.session_state.plot_const = {
            "trend_types": ["", "Linear", "Smooth LOWESS Curve"],
            "centile_types": ["", "CN", "CN_Males", "CN_Females", "CN_ICV_Corrected"],
            "linfit_trace_types": ["lin_fit", "conf_95%"],
            "centile_trace_types": [
                "centile_5",
                "centile_25",
                "centile_50",
                "centile_75",
                "centile_95",
            ],
            "distplot_trace_types": ["histogram", "density", "rug"],
            "min_per_row": 1,
            "max_per_row": 5,
            "num_per_row": 3,
            "margin": 20,
            "h_init": 500,
            "h_coeff": 1.0,
            "h_coeff_max": 2.0,
            "h_coeff_min": 0.6,
            "h_coeff_step": 0.2,
            "distplot_binnum": 100,
        }

        # Plot variables
        st.session_state.plot_var = {
            "df_data": pd.DataFrame(),
            "hide_settings": False,
            "hide_legend": False,
            "show_img": False,
            "plot_type": "Scatter Plot",
            "xvar": "",
            "xmin": -1.0,
            "xmax": -1.0,
            "yvar": "",
            "ymin": -1.0,
            "ymax": -1.0,
            "hvar": "",
            "hvals": [],
            "corr_icv": False,
            "plot_cent_normalized": False,
            "trend": "Linear",
            "traces": ["data", "lin_fit"],
            "lowess_s": 0.5,
            "centtype": "",
            "h_coeff": 1.0,
        }
        ###################################

        ###################################
        # Color maps for plots
        st.session_state.plot_colors = {
            "data": px.colors.qualitative.Set1,
            "centile": [
                "rgba(0, 0, 120, 0.5)",
                "rgba(0, 0, 90, 0.7)",
                "rgba(0, 0, 60, 0.9)",
                "rgba(0, 0, 90, 0.7)",
                "rgba(0, 0, 120, 0.5)",
            ],
        }
        ###################################

        ###################################
        # MRI view
        st.session_state.mriview_const = {
            "img_views": ["axial", "coronal", "sagittal"],
            "w_init": 500,
            "w_coeff": 1.0,
            "w_coeff_max": 2.0,
            "w_coeff_min": 0.6,
            "w_coeff_step": 0.2,
        }

        st.session_state.mriview_var = {
            "crop_to_mask": True,
            "show_overlay": True,
            "list_orient": ["axial", "coronal", "sagittal"],
            "w_coeff": 1.0,
        }

        ###################################

        ###################################
        # ROI dictionaries
        # List of roi names, indices, etc.
        st.session_state.rois = {
            "path": os.path.join(st.session_state.paths["root"], "resources", "lists"),
            "roi_dict_options": [
                "",
                "muse_rois",
            ],  # This will be extended with additional roi dict.s
            "roi_csvs": {
                "muse_rois": "MUSE_listROIs.csv",
                "muse_derived": "MUSE_mapping_derivedROIs.csv",
            },
            "sel_roi_dict": "muse_rois",
            "sel_derived_dict": "muse_derived",
        }

        # Read initial roi lists (default:MUSE) to dictionaries
        ssroi = st.session_state.rois
        df_tmp = pd.read_csv(
            os.path.join(ssroi["path"], ssroi["roi_csvs"][ssroi["sel_roi_dict"]])
        )
        dict1 = dict(zip(df_tmp["Index"].astype(str), df_tmp["Name"].astype(str)))
        dict2 = dict(zip(df_tmp["Name"].astype(str), df_tmp["Index"].astype(str)))
        dict3 = utilroi.muse_derived_to_dict(
            os.path.join(ssroi["path"], ssroi["roi_csvs"][ssroi["sel_derived_dict"]])
        )
        st.session_state.rois["roi_dict"] = dict1
        st.session_state.rois["roi_dict_inv"] = dict2
        st.session_state.rois["roi_dict_derived"] = dict3
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
        st.session_state.sel_roi_img = ""

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
            st.session_state.paths["experiment"],
            st.session_state.dict_paths[d_tmp][0],
            st.session_state.dict_paths[d_tmp][1],
        )
        print(f"setting {st.session_state.paths[d_tmp]}")

    st.session_state.paths["dlmuse_csv"] = os.path.join(
        st.session_state.paths["dlmuse"], "DLMUSE_Volumes.csv"
    )

    st.session_state.paths["csv_mlscores"] = os.path.join(
        st.session_state.paths["mlscores"],
        f"{st.session_state.experiment}_DLMUSE+MLScores.csv",
    )

    st.session_state.paths["demog_csv"] = os.path.join(
        st.session_state.paths["experiment"], "lists", "Demog.csv"
    )

    st.session_state.paths["csv_plot"] = os.path.join(
        st.session_state.paths["plots"], "Data.csv"
    )
    # Reset plot data
    st.session_state.plot_var["df_data"] = pd.DataFrame()


def reset_flags() -> None:
    """
    Resets flags if the working dir changed
    """
    for tmp_key in st.session_state.flags.keys():
        st.session_state.flags[tmp_key] = False
    st.session_state.flags["experiment"] = True

    # Check dicom folder
    fcount = utilio.get_file_count(st.session_state.paths["dicoms"])
    if fcount > 0:
        st.session_state.flags["dicoms"] = True


def reset_plots() -> None:
    """
    Reset plot variables when data file changes
    """
    st.session_state.plots = pd.DataFrame(columns=st.session_state.plots.columns)
    st.session_state.plot_sel_vars = []
    st.session_state.plot_var["hide_settings"] = False
    st.session_state.plot_var["hide_legend"] = False
    st.session_state.plot_var["show_img"] = False
    st.session_state.plot_var["plot_type"] = False
    st.session_state.plot_var["xvar"] = ""
    st.session_state.plot_var["xmin"] = -1.0
    st.session_state.plot_var["xmax"] = -1.0
    st.session_state.plot_var["yvar"] = ""
    st.session_state.plot_var["ymin"] = -1.0
    st.session_state.plot_var["ymax"] = -1.0
    st.session_state.plot_var["hvar"] = ""
    st.session_state.plot_var["hvals"] = []
    st.session_state.plot_var["corr_icv"] = False
    st.session_state.plot_var["plot_cent_normalized"] = False
    st.session_state.plot_var["trend"] = "Linear"
    st.session_state.plot_var["traces"] = ["data", "lin_fit"]
    st.session_state.plot_var["lowess_s"] = 0.5
    st.session_state.plot_var["centtype"] = ""
    st.session_state.plot_var["h_coeff"] = 1.0
