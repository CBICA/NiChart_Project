import os
import shutil
from typing import Any

import jwt
import time
import pandas as pd
import plotly.express as px
import streamlit as st
import utils.utils_io as utilio
import utils.utils_rois as utilroi
import utils.utils_processes as utilproc
from PIL import Image

# from streamlit.web.server.websocket_headers import _get_websocket_headers

def init_paths():
    # Set paths to data models, etc.

    # Set default directories
    p_root = os.path.dirname(os.path.dirname(os.getcwd()))
    p_init = p_root
    p_resources = os.path.join(
        p_root, "resources"
    )
    p_centiles = os.path.join(
        p_resources, "reference_data", "centiles"
    )
    p_proc_def = os.path.join(
        p_resources, "process_definitions"
    )
    
    # Set output path
    user_id = ''
    if st.session_state.has_cloud_session:
        user_id = st.session_state.cloud_user_id
    p_out = os.path.join(
        p_root, 'output_folder', user_id
    )
    if not os.path.exists(p_out):
        os.makedirs(p_out)

    st.session_state.dicts = {
        "muse_derived": os.path.join(
            p_resources, "MUSE", "list_MUSE_mapping_derived.csv"
        ),
        "muse_all": os.path.join(p_resources, "MUSE", "list_MUSE_all.csv"),
        "muse_sel": os.path.join(p_resources, "MUSE", "list_MUSE_primary.csv"),
    }

    st.session_state.paths = {
        "root": p_root,
        "init": p_init,
        "resources": p_resources,
        "centiles" : p_centiles,
        "proc_def": p_proc_def,
        "file_search_dir": "",
        "out_dir": p_out,
        "project": "",
    }

def init_selections() -> None:
    st.session_state.selections = {
        'sel_roi_group' : 'Primary',
        'sel_roi' : 'GM',
    }


def init_plot_vars() -> None:
    ###################################
    # Plotting
    # Dictionary with plot info
    st.session_state.plots = pd.DataFrame(columns=['params'])
    st.session_state.plot_curr = -1

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
        "num_per_row": 2,
        "margin": 20,
        "h_init": 500,
        "h_coeff": 1.0,
        "h_coeff_max": 2.0,
        "h_coeff_min": 0.6,
        "h_coeff_step": 0.2,
        "distplot_binnum": 100,
    }

    # Plot data
    st.session_state.curr_df = pd.DataFrame()

    # Plot variables
    st.session_state.plot_params = {
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
        "centile_type": 'CN',
        "centile_values": ['centile_25', 'centile_50', 'centile_75'],
        "h_coeff": 1.0,
        "ptype": 'scatter'
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
    
def init_pipeline_definitions() -> None:
    plist = os.path.join(
        st.session_state.paths['resources'], 'pipelines', 'list_pipelines.csv'
    )
    st.session_state.pipelines = pd.read_csv(plist)
    
    print(st.session_state.pipelines)

def init_reference_data() -> None:
    indir = os.path.join(
        st.session_state.paths['resources'], 'reference_data', 'sample1'
    )
    t1 = os.path.join(indir, 't1', 'sample1_T1.nii.gz')
    fl = os.path.join(indir, 'fl', 'sample1_FL.nii.gz')
    dlmuse = os.path.join(indir, 'dlmuse', 'sample1_T1_DLMUSE.nii.gz')
    dlwmls = os.path.join(indir, 'dlwmls', 'sample1_FL_DLWMLS.nii.gz')
    st.session_state.ref_data = {
        't1' : t1,
        'fl' : fl,
        'dlmuse' : dlmuse,
        'dlwmls' : dlwmls
    }

def init_muse_roi_def() -> None:
    # Paths to roi lists
    muse = {
        'path': os.path.join(st.session_state.paths['resources'], 'lists', 'MUSE'),
        'list_rois' : 'MUSE_listROIs.csv',
        'list_derived' : 'MUSE_mapping_derivedROIs.csv',
        'list_groups' : 'MUSE_ROI_Groups_v1.csv'
    }
    
    # Read roi lists to dictionaries
    df_tmp = pd.read_csv(
        os.path.join(muse['path'], muse['list_rois'])
    )
    dict1 = dict(zip(df_tmp["Index"].astype(str), df_tmp["Name"].astype(str)))
    dict2 = dict(zip(df_tmp["Name"].astype(str), df_tmp["Index"].astype(str)))
    dict3 = utilroi.muse_derived_to_dict(
        os.path.join(muse['path'], muse['list_derived'])
    )
    df_derived = utilroi.muse_derived_to_df(
        os.path.join(muse['path'], muse['list_derived'])
    )
    df_groups = utilroi.muse_roi_groups_to_df(
        os.path.join(muse['path'], muse['list_groups'])
    )
    muse['dict_roi'] = dict1
    muse['dict_roi_inv'] = dict2
    muse['dict_derived'] = dict3
    muse['df_derived'] = df_derived
    muse['df_groups'] = df_groups
    
    # Read MUSE ROI lists
    st.session_state.rois = {
        'muse' : muse
    }

def update_default_paths() -> None:
    """
    Update default paths in session state if the working dir changed
    """
    print('FIXME')

def reset_flags() -> None:
    """
    Resets flags if the working dir changed
    """
    for tmp_key in st.session_state.flags.keys():
        st.session_state.flags[tmp_key] = False
    st.session_state.flags["project"] = True

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

def update_process_def(sel_dir) -> None:
    """
    Updates process definitions
    """
    if sel_dir is None:
        return

    sel_dir = os.path.abspath(sel_dir)

    # Update process data
    steps = utilproc.load_steps_from_yaml(st.session_state.paths["proc_def"])

    # Create process graph
    graph = utilproc.build_graph(steps)

    # Detect file roles
    file_roles = utilproc.get_file_roles(steps)
    
    # Exclude files that are outputs of any step (only allow true source files)
    out_files = {f for step in steps.values() for f in step['out_list']}
    in_files = sorted(set(file_roles.keys()) - out_files)
    
    st.session_state.processes['steps'] = steps
    st.session_state.processes['graph'] = graph
    st.session_state.processes['roles'] = file_roles
    st.session_state.processes['in_files'] = in_files
    st.session_state.processes['out_files'] = out_files
    st.session_state.processes['sel_inputs'] = []
    st.session_state.processes['sel_steps'] = []

def update_out_dir(sel_outdir) -> None:
    """
    Updates when outdir changes
    """
    if sel_outdir is None:
        return

    if sel_outdir == st.session_state.paths['out_dir']:
        return

    sel_outdir = os.path.abspath(sel_outdir)
    if not os.path.exists(sel_outdir):
        try:
            os.makedirs(sel_outdir)
        except:
            st.error(f'Could not create folder: {sel_outdir}')
            return

    # Set out dir path
    st.session_state.paths['out_dir'] = sel_outdir
    st.session_state.flags["out_dir"] = True

    # Reset other vars
    st.session_state.navig['project'] = None

def update_project(sel_project) -> None:
    """
    Updates when outdir changes
    """
    if sel_project is None:
        return

    if sel_project == st.session_state.navig['project']:
        return

    # Create project dir
    project_dir = os.path.join(
        st.session_state.paths['out_dir'],
        sel_project
    )
    
    try:
        if not os.path.exists(project_dir):
            os.makedirs(project_dir)
            #st.success(f'Created folder {project_dir}')
            #time.sleep(3)
    except:
        st.error(f'Could not create project folder: {project_dir}')
        return

    # Set project name
    st.session_state.navig['project'] = sel_project
    #st.session_state.flags["project"] = True
    st.session_state.paths['project'] = project_dir
    st.session_state.paths['project_curr_path'] = project_dir

    # Reset other vars
    update_default_paths()

def config_page() -> None:
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
        return ""
    return headers["X-Amzn-Oidc-Data"]

def process_session_user_id() -> Any:
    headers = st.context.headers
    if not headers or "X-Amzn-Oidc-Identity" not in headers:
        return "NO_USER_FOUND"
    return headers["X-Amzn-Oidc-Identity"]

def process_session_user_email() -> Any:
    headers = st.context.headers
    if not headers or "X-Amzn-Oidc-Data" not in headers:
        return "NO_EMAIL_FOUND"
    raw_token = headers['X-Amzn-Oidc-Data']
    decoded_token = jwt.decode(
        raw_token,
        algorithms=["ES256"],
        options={"verify_signature": False},
    )
    if not decoded_token or 'email' not in decoded_token:
        return "NO_EMAIL_FOUND"
    return decoded_token['email']

def init_session_state() -> None:
    # Initiate Session State Values
    if "instantiated" not in st.session_state:
        
        ####################################
        ### Page settings
        
        # App icon image
        st.session_state.nicon = Image.open("../resources/nichart1.png")

        st.session_state.sel_pipeline = None

        # Menu navigation
        st.session_state.sel_menu = 'Home'

        st.session_state.user = {
            'setup_sel_item': None,
            'setup_project_update': False,
            'setup_project_mode': 0,
        }


        st.session_state.navig = {
            'main_menu': "Home",
            'workflow': None,
            'pipeline_step': None,
            'project': None
        }

        ####################################
        # Settings specific to desktop/cloud
        
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

        # Store user session info for later retrieval
        if st.session_state.app_type == "cloud":
            st.session_state.cloud_session_token = process_session_token()
            if st.session_state.cloud_session_token:
                st.session_state.has_cloud_session = True
                st.session_state.cloud_user_id = process_session_user_id()
                st.session_state.cloud_user_email = process_session_user_email()
            else:
                st.session_state.has_cloud_session = False
        else:
            st.session_state.has_cloud_session = False

        ####################################
        # Initialize paths
        init_paths()


        # Set default project
        sel_project = 'Experiment_1'
        update_project(sel_project)
 """        ###################################
        # Pipelines
        st.session_state.pipelines = [
            "Home",
            "sMRI Biomarkers (T1)",
            "WM Lesion Segmentation (FL)",
            "DTI Biomarkers (DTI)",
            "Resting State fMRI Biomarkers (rsfMRI)",
            "Sample Container Workflow"
        ]
        st.session_state.pipeline = "Home"
        st.session_state._pipeline = st.session_state.pipeline
        ###################################

        ###################################
        # General
        # Study name
        st.session_state.dset = ""

        # Icons for panels
        st.session_state.icon_thumb = {
            False: ":material/thumb_down:",
            True: ":material/thumb_up:",
        }

        # Flags for checkbox states
        st.session_state.checkbox = {
            "dicoms_wdir": False,
            "dicoms_in": False,
            "dicoms_series": False,
            "dicoms_run": False,
            "dicoms_view": False,
            "dicoms_download": False,
            "dlmuse_wdir": False,
            "dlmuse_in": False,
            "dlmuse_run": False,
            "dlmuse_view": False,
            "dlmuse_download": False,
            "dlwmls_wdir": False,
            "dlwmls_in": False,
            "dlwmls_run": False,
            "dlwmls_view": False,
            "dlwmls_download": False,
            "ml_wdir": False,
            "ml_inrois": False,
            "ml_indemog": False,
            "ml_run": False,
            "ml_download": False,
            "view_wdir": False,
            "view_in": False,
            "view_select": False,
            "view_plot": False,
        }

        # Flags for various i/o
        st.session_state.flags = {
            "dset": False,
            "dir_out": False,
            "dir_dicom": False,
            "dicom_series": False,
            "dir_nifti": False,
            "dir_t1": False,
            "dir_dlmuse": False,
            "csv_dlmuse": False,
            "csv_dlwmls": False,
            "csv_demog": False,
            "csv_dlmuse+demog": False,
            "dir_download": False,
            "csv_mlscores": False,
            "csv_plot": False,
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
        } """

        # Set initial values for paths
        st.session_state.paths["root"] = os.path.dirname(os.path.dirname(os.getcwd()))
        st.session_state.paths["init"] = st.session_state.paths["root"]
        if st.session_state.has_cloud_session:
            user_id = st.session_state.cloud_user_id
            st.session_state.paths["dir_out"] = os.path.join(
                #st.session_state.paths["root"], "output_folder", user_id
                "/fsx", user_id
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
                    st.session_state.paths["out_dir"], demo_name
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

        ####################################
        # Image modalities
        st.session_state.list_mods = [
            "T1", "T2", "FL", "DTI", "fMRI"
        ]

        init_muse_roi_def()
        init_pipeline_definitions()
        init_reference_data()
        init_plot_vars()
        init_selections()

        ####################################
        # Process definitions
        # Used to keep process info provided in yaml files
        st.session_state.processes = {
            'steps': None,
            'roles': None,
            'in_files': None,
            'out_files': None,
            'sel_inputs': [],
            'sel_steps': [],
        }
        #update_process_def(st.session_state.paths['proc_def'])

        
        ####################################
        # Various parameters

        # Average ICV estimated from a large sample
        # IMPORTANT: Used in NiChart Engine for normalization!
        st.session_state.params = {
            'mean_icv': 1430000,
            'harm_min_samples': 30,
        }
        
        st.session_state._sel_step1 = None
        
        # Icons for panels
        
        ####################################
        # Miscallenous settings
        st.session_state.misc = {
            'icon_thumb': {
                False: ":material/thumb_down:",
                True: ":material/thumb_up:",
            }
        }

        st.session_state.instantiated = True

