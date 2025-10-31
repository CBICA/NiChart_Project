import os
import shutil
from typing import Any

import jwt
import time
import yaml
import pandas as pd
import streamlit as st
import utils.utils_rois as utilroi
import utils.utils_processes as utilproc
import utils.utils_cmaps as utilcmap
import os
from PIL import Image
import streamlit_antd_components as sac

# from streamlit.web.server.websocket_headers import _get_websocket_headers

def disp_selections():
    '''
    Show user selections
    '''
    #with st.sidebar:
        #sac.divider(label='Selections', icon = 'person', align='center', color='gray')
        #if st.session_state.project is not None:
            #st.markdown(f'`Project Name: {st.session_state.project}`')
        #if st.session_state.sel_pipeline is not None:
            #st.markdown(f'`Pipeline: {st.session_state.sel_pipeline}`')
    print('FIXME: This is bypassed for now ...')
    
def disp_session_state():
    '''
    Show session state variables
    '''
    if '_debug_flag_show' not in st.session_state:
        st.session_state['_debug_flag_show'] = st.session_state['debug']['flag_show']

    def update_val():
        st.session_state['debug']['flag_show'] = st.session_state['_debug_flag_show']

    sac.divider(label='Debug', icon = 'gear',  align='center', color='gray')
    st.checkbox(
        'Show Session State',
        key = '_debug_flag_show',
        on_change = update_val
    )
    
    if st.session_state['debug']['flag_show']:
        with st.container(border=True):
            st.markdown('##### Session State:')
            list_items = sorted([x for x in st.session_state.keys() if not x.startswith('_')])
            st.pills(
                "Select Session State Variable(s) to View",
                list_items,
                selection_mode="multi",
                key='_debug_sel_vars',
                default=st.session_state['debug']['sel_vars'],
                label_visibility="collapsed",
            )
            st.session_state['debug']['sel_vars'] = st.session_state['_debug_sel_vars']

            for sel_var in st.session_state['debug']['sel_vars']:
                st.markdown('➤ ' + sel_var + ':')
                st.write(st.session_state[sel_var])
    print('FIXME: This is bypassed for now ...')

def init_project_folders():
    '''
    Set initial values for project folders
    '''
    dnames = [
        "t1", "fl", "participants", "dlmuse_seg", "dlmuse_vol"
    ]
    dtypes = [
        "in_img", "in_img", "in_csv", "out_img", "out_csv"
    ]
    st.session_state.project_folders = pd.DataFrame(
        {"dname": dnames, "dtype": dtypes}
    )

def init_session_vars():
    '''
    Set initial values for session variables
    '''
    ## Misc variables
    # st.session_state.mode = 'release'
    st.session_state.mode = 'debug'

    st.session_state.skip_survey = True

    st.session_state.sel_add_button = None

    #st.session_state.project = 'nichart_project'
    st.session_state.project = 'user_default'
    
    st.session_state.sel_pipeline = 'dlmuse'

    st.session_state.sel_mrid = None
    st.session_state.sel_roi = None

    st.session_state.pipeline_colors = [
        'red', 'pink', 'grape', 'violet', 'indigo', 'blue',
        'cyan', 'teal', 'green', 'lime', 'yellow', 'orange',
    ]

    st.session_state.list_mods = ["T1", "T2", "FL", "DTI", "fMRI"]
    st.session_state.params = {
        'mean_icv': 1430000,  # Average ICV estimated from a large sample
        'harm_min_samples': 30,
    }
    st.session_state.misc = {
        'icon_thumb': {         # Icons for panels
            False: ":material/thumb_down:",
            True: ":material/thumb_up:",
        }
    }

    ## Debug vars
    st.session_state['debug'] = {
        'flag_show': False,
        'sel_vars': []
    }

    ## Page settings

    # App icon image
    st.session_state.nicon = Image.open("../resources/nichart1.png")

    # Menu navigation
    st.session_state.sel_menu = 'Home'

    # User info
    st.session_state.user = {
        'setup_sel_item': None,
        'setup_project_update': False,
        'setup_project_mode': 0,
    }

    ####################################
    ### Settings specific to desktop/cloud
    
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
        else:
            st.session_state.has_cloud_session = False
    else:
        st.session_state.has_cloud_session = False

def copy_test_folders():
    '''
    Copy demo folders into user folders as needed
    '''
    if st.session_state.has_cloud_session:
        # Copy demo dirs to user folder (TODO: make this less hardcoded)
        demo_dir_paths = [
            os.path.join(
                st.session_state.paths["root"],
                "output_folder",
                "NiChart_Demo1",
            ),
            os.path.join(
                st.session_state.paths["root"],
                "output_folder",
                "NiChart_Demo2",
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

def init_paths():
    '''
    Set paths to pre-defined folders
    '''
    # Resources
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
    
    # Output
    user_id = ''
    if st.session_state.has_cloud_session:
        user_id = st.session_state.cloud_user_id
        p_out = os.path.join(
            "/fsx/fsx/", user_id
        )
    else:
        p_out = os.path.join(
            p_root, 'output_folder', user_id
        )
    if not os.path.exists(p_out):
        os.makedirs(p_out)
    
    # Paths specific to project
    p_prj = os.path.join(
        p_out, st.session_state.project
    )
    if not os.path.exists(p_prj):
        os.makedirs(p_prj)

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
        "host_out_dir": None,
        "project": p_prj,
        'target': None
    }

    # Host-container dir mapping which can be useful for local nested containers
    # Code which relies on this should always check if it is None
    # And use local paths instead if so.
    host_out_dir = os.getenv("NICHART_HOST_DATA_DIR", None)
    if host_out_dir is not None:
        st.session_state.paths['host_out_dir'] = host_out_dir
    
    # List of output folders
    st.session_state.out_dirs = [
        'participants',
        'dicoms',
        't1', 't2', 'fl', 'fmri', 'dti',
        'dlmuse_seg', 'dlmuse_vol',
        'dlwmls', 'spare',
    ]
    
    ############
    # FIXME : set init folder to test folder outside repo
    st.session_state.paths["init"] = os.path.join(
        st.session_state.paths["root"], "test_data"
    )
    st.session_state.paths["file_search_dir"] = st.session_state.paths["init"]
    ############    

def init_dicom_vars() -> None:
    '''
    Set dicom variables
    '''
    st.session_state.dicoms = {
        'list_series': None,
        'num_dicom_scans': 0,
        'df_dicoms': None
    }

def init_plot_vars() -> None:
    '''
    Set plotting variables
    '''
    # Dataframe that keeps parameters for all plots
    st.session_state.plots = pd.DataFrame(columns=['flag_sel', 'params'])
    st.session_state.plot_curr = -1

    st.session_state.plot_active = None


    # Plot data
    st.session_state.plot_data = {
        'df_data': None,
        'df_cent': None
    }

    # Plot settings
    st.session_state.plot_settings = {
        "flag_hide_settings": 'Show',
        "flag_hide_legend": 'Show',
        "flag_hide_mri": 'Show',
        "trend_types": ["None", "Linear", "Smooth LOWESS Curve"],
        "centile_types": ["", "CN", "CN_Males", "CN_Females", "CN_ICV_Corrected"],
        "linfit_trace_types": [
            "lin_fit", "conf_95%"
        ],
        "centile_trace_types": [
            "centile_5", "centile_25", "centile_50", "centile_75", "centile_95",
        ],
        "distplot_trace_types": [
            "histogram", "density", "rug"
        ],
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
        "cmaps": utilcmap.cmaps_init,
        "alphas": utilcmap.alphas_init,
        "w_centile": 6,
        "w_fit": 6,
        "min_age": 20,
        "max_age": 100,
        #"cmaps2": utilcmap.cmaps2,
        #"cmaps3": utilcmap.cmaps3,
    }

    # Plot parameters specific to each plot
    st.session_state.plot_params = {
        "plot_type": "scatter",
        "xvargroup": 'demog',
        "xvar": 'Age',
        "xmin": None,
        "xmax": None,
        "yvargroup": 'MUSE_ShortList',
        "yvar": 'GM',
        "ymin": None,
        "ymax": None,
        "hvargroup": 'cat_vars',
        "hvar": None,
        "hvals": None,
        "fvargroup": 'cat_vars',
        "fvar": None,
        "fvals": None,
        "corr_icv": False,
        "plot_cent_normalized": False,
        "trend": "Linear",
        "show_conf": True,
        "traces": None,
        "lowess_s": 0.7,
        "centile_type": 'CN',
        "centile_values": ['centile_50'],
        "flag_norm_centiles": False,
        "list_roi_indices": [81, 82],
        "list_orient": ["axial", "coronal", "sagittal"],
        "is_show_overlay": True,
        "crop_to_mask": True,
        'filter_sex': ['F', 'M'],
        'filter_age': [40, 95],
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

def init_var_groups() -> None:
    '''
    Read variable groups to a dataframe
    '''
    f_vars = os.path.join(
        st.session_state.paths['resources'], 'lists', 'dict_var_groups.yaml'
    )

    with open(f_vars, 'r') as file:
        data = yaml.safe_load(file)

    rows = []
    for group_name, group_info in data.items():
        raw_values = group_info.get('values', [])
        str_values = [str(v) for v in raw_values]  # ensure uniform type
        rows.append({
            'group': group_name,
            'category': group_info.get('category'),
            'vtype': group_info.get('vtype'),
            'atlas': group_info.get('atlas'),
            'values': str_values
        })

    df = pd.DataFrame(rows)
    st.session_state.dicts['df_var_groups'] = df

def init_dicts() -> None:
    '''
    Initialize all data dictionaries (atlas roi def.s etc.)
    '''
    # MUSE dictionaries
    muse = utilroi.read_muse_dicts()
    st.session_state.dicts = {
        'muse': muse
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
        os.path.join(muse['path'], muse['list_rois']),
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

def update_project(sel_project) -> None:
    """
    Updates when project changes
    """
    if sel_project is None:
        return

    if sel_project == st.session_state.project:
        return

    # Create project dir
    p_prj = os.path.join(
        st.session_state.paths['out_dir'], sel_project
    )

    try:
        if not os.path.exists(p_prj):
            os.makedirs(p_prj)
            st.success(f'Created folder {p_prj}')
            time.sleep(1)
    except:
        st.error(f'Could not create project folder: {p_prj}')
        return

    # Set project name
    st.session_state.project = sel_project
    st.session_state.paths['project'] = p_prj

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
        
        # Set initial session variables
        init_session_vars()

        # Set output files
        init_project_folders()

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

        # Initialize dicts
        init_dicts()

        # Initialize variable groups
        init_var_groups()

        # FIXME : set init folder to test folder outside repo
        st.session_state.paths["init"] = os.path.join(
            st.session_state.paths["root"], "test_data"
        )
        st.session_state.paths["file_search_dir"] = st.session_state.paths["init"]

        # Update project variables
        update_project(st.session_state.project)

        # Copy test data to user folder
        copy_test_folders

        # Init variables for different pages 
        init_muse_roi_def()
        init_pipeline_definitions()
        init_reference_data()
        init_plot_vars()
        init_dicom_vars()
        
        # Set flag
        st.session_state.instantiated = True

