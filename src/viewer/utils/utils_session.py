import os
import shutil
from typing import Any
import re

import jwt
import time
import yaml
import pandas as pd
import streamlit as st
import utils.utils_rois as utilroi
import utils.utils_processes as utilproc
import utils.utils_cmaps as utilcmap
import utils.utils_toolloader as utiltl
import utils.utils_upload as utilup
import os
from PIL import Image
import streamlit_antd_components as sac

# from streamlit.web.server.websocket_headers import _get_websocket_headers
    
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
            #list_items = sorted([x for x in st.session_state.keys() if x.startswith('_')])
            st.pills(
                "Select Session State Variable(s) to View",
                list_items,
                selection_mode="multi",
                key='_debug_sel_vars',
                # default=st.session_state['debug']['sel_vars'],
                label_visibility="collapsed",
            )
            st.session_state['debug']['sel_vars'] = st.session_state['_debug_sel_vars']

            for sel_var in st.session_state['debug']['sel_vars']:
                st.markdown('➤ ' + sel_var + ':')
                st.write(st.session_state[sel_var])
    #print('FIXME: This is bypassed for now ...')

def expand_values(raw_values: list, d_lists: dict) -> list:
    '''
    Expand a list with pattern [{listname}] or [prefix_{listname}].
    Returns expanded values.
    '''
    values = []
    for item in raw_values:
        s = str(item)
        m = re.fullmatch(r'(.*?)\{([^}]+)\}(.*)', s)
        if m:
            prefix, list_name, suffix = m.group(1), m.group(2), m.group(3)
            list_entry = d_lists.get(list_name, {})
            list_vals = list_entry.get('values', [])
            for v in list_vals:
                sv = str(v)
                values.append(f"{prefix}{sv}{suffix}")

        else:
            values.append(s)
    return values

def init_folders():
    '''
    Set initial values for project folders
    '''
    dnames = [
        "t1", "fl", "participants", "dlmuse_seg", "dlmuse_vol"
    ]
    dtypes = [
        "in_img", "in_img", "in_csv", "out_img", "out_csv"
    ]
    st.session_state.prj_folders = pd.DataFrame(
        {"dname": dnames, "dtype": dtypes}
    )
    
def init_scan():
    '''
    Set scan info
    '''
    st.session_state.curr_scan = None

def init_participant():
    '''
    Set participant info
    '''
    st.session_state.participant = {
        'mrid' : None,
        'age' : None,
        'sex' : None,
    }

def init_user():
    '''
    Set user info
    '''
    st.session_state.user = {
        'setup_sel_item': None,
        'setup_project_update': False,
        'setup_project_mode': 0,
    }

def init_cloud():
    '''
    Set cloud vars
    '''
    app_type = "desktop"
    has_cloud_session = False
    forced_cloud = False
    cloud_session_token = None    
    cloud_user_id = None
    cloud_user_email = None

    if os.getenv("NICHART_FORCE_CLOUD", "0") == "1":
        forced_cloud = True
        app_type = "cloud"
        cloud_session_token = process_session_token()
        if cloud_session_token:
            has_cloud_session = True
            cloud_user_id = process_session_user_id()
            cloud_user_email = process_session_user_email()    
    
    app_config = {
        "cloud": {"msg_infile": "Upload"},
        "desktop": {"msg_infile": "Select"},
    }

    tdict = {
        'app_type': app_type,
        'has_cloud_session': has_cloud_session,
        'forced_cloud': forced_cloud,
        'cloud_session_token': cloud_session_token,
        'cloud_user_id': cloud_user_id,
        'cloud_user_email': cloud_user_email,
        'app_config': app_config
    }
    st.session_state.update(tdict)

def init_pipelines():
    '''
    Set pipeline info
    '''
    pipeline_cat = utiltl.overall_pipeline_category_listing()
    pipeline_req = utiltl.overall_pipeline_requirements_listing()
    pipeline_colors = [
        'red', 'pink', 'grape', 'violet', 'indigo', 'blue',
        'cyan', 'teal', 'green', 'lime', 'yellow', 'orange',
    ]

    tdict = {
        'pipeline_categories': pipeline_cat,
        'pipeline_requirements': pipeline_req,
        'harmonizable_pipelines': pipeline_cat['harmonized'],
        'pipeline_colors': pipeline_colors,
        'sel_pipeline': None
    }
    st.session_state.update(tdict)

def init_misc():
    '''
    Set initial values for session variables
    '''
    tdict = {
        'mode': 'debug',
        'show_settings': False,
        'layout_plots': 'Sidebar',
        'skip_survey': True,
        'workflow': None,
        'sel_add_button': None,
        'sel_mrid': None,
        'sel_roi': None,
        'do_harmonize': False,
        'nifti_dicom_upload_mode': None,
        'list_mods': ["T1", "T2", "FL", "DTI", "fMRI", "T1CE", "ADC"],
        'params': {
            'mean_icv': 1430000,  # Average ICV estimated from a large sample
            'harm_min_samples': 30,
        },
        'misc': {
            'icon_thumb': {         # Icons for panels
                False: ":material/thumb_down:",
                True: ":material/thumb_up:",
            },
        },
        'debug': {
            'flag_show': False,
            'sel_vars': [],
        },
        'nicon': Image.open("../resources/nichart1.png"),
        'sel_menu': 'Home',

    }
    st.session_state.update(tdict)

def init_paths():
    '''
    Set paths to pre-defined folders
    '''
    # Set paths for resources
    p_root = os.path.dirname(os.path.dirname(os.getcwd()))
    p_resources = os.path.join(p_root, "resources")
    p_centiles = os.path.join(p_resources, "reference_data", "centiles")
    p_sample = os.path.join(p_root, "sample_datasets", "demo_dataset")    
    p_proc_def = os.path.join(p_resources, "process_definitions")
    p_init = p_root
    
    # Set path for user output folder
    user_id = ''
    if st.session_state.has_cloud_session:
        user_id = st.session_state.cloud_user_id
        p_out = os.path.join(
            "/fsx/fsx/", user_id
        )
    else:
        p_out = os.path.join(p_root, 'output_folder', user_id)
    os.makedirs(p_out, exist_ok=True)
    
    st.session_state.paths = {
        "root": p_root,
        "init": p_init,
        "resources": p_resources,
        "sample_data": p_sample,
        "centiles" : p_centiles,
        "proc_def": p_proc_def,
        "file_search_dir": "",
        "out_dir": p_out,
        "host_out_dir": None,
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
        't1', 't2', 'fl', 'fmri', 'dti',
        'dlmuse_seg', 'dlmuse_vol',
        'dlwmls_seg', 'dlwmls_vol',
        'ml_biomarkers', 'ravens', 'opnmf',
        'plot_data', 'reports'
    ]
    
    ############
    # FIXME : set init folder to test folder outside repo
    st.session_state.paths["init"] = os.path.join(
        st.session_state.paths["root"], "test_data"
    )
    st.session_state.paths["file_search_dir"] = st.session_state.paths["init"]
    ############    

def init_dicoms() -> None:
    '''
    Init dicom variables
    '''
    st.session_state.dicoms = {
        'list_series': None,
        'sel_serie': None,
        'num_dicom_scans': 0,
        'df_dicoms': None
    }
    
def init_pipeline_definitions() -> None:
    plist = os.path.join(
        st.session_state.paths['resources'], 'pipelines', 'list_pipelines.csv'
    )
    st.session_state.pipelines = pd.read_csv(plist)
    
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
        pipeline = group_info.get('pipeline', [])
        if pipeline is None:
            pipeline = []
        rows.append({
            'group': group_name,
            'category': group_info.get('category'),
            'pipeline': pipeline,
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
    st.session_state.dicts = {}

    # Read MUSE dictionaries
    utilroi.muse_read_dicts()    

    # Read roi lists info
    f_in = os.path.join(
        st.session_state.paths['resources'], 'dicts', 'dict_roi_lists.yaml'
    )
    with open(f_in, 'r') as file:
        data = yaml.safe_load(file)
    st.session_state.dicts['roi_lists'] = data

    # Read project files info
    f_in = os.path.join(
        st.session_state.paths['resources'], 'dicts', 'dict_project_files.yaml'
    )
    with open(f_in, 'r') as file:
        data = yaml.safe_load(file)
    st.session_state.dicts['project_files'] = data

    # Read project vars info
    f_in = os.path.join(
        st.session_state.paths['resources'], 'dicts', 'dict_project_vars.yaml'
    )
    with open(f_in, 'r') as file:
        data = yaml.safe_load(file)

    # Expand lists in project variable definitions
    for vkey, vval in data.items():
        raw_vars = vval.get('vars', [])
        exp_vars = expand_values(raw_vars, st.session_state.dicts['roi_lists'])
        data[vkey]['vars'] = exp_vars
    st.session_state.dicts['project_vars'] = data

    # Read var renaming info
    f_in = os.path.join(
        st.session_state.paths['resources'], 'dicts', 'dict_var_renaming.yaml'
    )
    with open(f_in, 'r') as file:
        data = yaml.safe_load(file)
    st.session_state.dicts['var_renaming'] = data


    # Read var groups info
    f_in = os.path.join(
        st.session_state.paths['resources'], 'dicts', 'dict_var_groups.yaml'
    )
    with open(f_in, 'r') as file:
        data = yaml.safe_load(file)
    st.session_state.dicts['var_groups'] = data

def init_project(sel_project) -> None:
    """
    Updates when project folder changes
    """
    if sel_project is None:
        return False

    # Update project folder only if user selects a new one
    if sel_project == st.session_state.get('prj_name', None):
        return False

    try:
        # Create project folder
        utilup.create_project_folder(sel_project)

        # Set session state variables
        p_prj = os.path.join(st.session_state.paths['out_dir'], sel_project)
        st.session_state.prj_name = sel_project
        st.session_state.paths['prj_dir'] = p_prj
        st.session_state.paths['curr_data'] = p_prj
        
        init_dicoms()
        init_scan()
        init_participant()

        return True

    except:
        return False

# Function to parse AWS login (if available)
def process_session_token() -> Any:
    # headers = _get_websocket_headers()
    headers = st.context.headers
    if not headers or "X-Amzn-Oidc-Data" not in headers:
        return ""
    return headers["X-Amzn-Oidc-Data"]

def process_session_access_token() -> Any:
    headers = st.context.headers
    if not headers or "X-Amzn-Oidc-Access-Token" not in headers:
        return ""
    return headers["X-Amzn-Oidc-Access-Token"]

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
    '''
    Initiate Session State Values
    '''
    if "instantiated" not in st.session_state:
        
        init_cloud()
        init_paths()
        init_project('user_default')
        init_pipelines()
        init_misc()
        init_folders()
        init_participant()
        init_scan()
        init_user()
        init_dicts()
        init_var_groups()
        init_pipeline_definitions()
        init_reference_data()
        init_dicoms()

        # Set flag
        st.session_state.instantiated = True

