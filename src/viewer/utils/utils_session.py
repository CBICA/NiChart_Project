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
    st.session_state.flags["task"] = True

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
    st.session_state.navig['task'] = None

def update_task(sel_task) -> None:
    """
    Updates when outdir changes
    """
    if sel_task is None:
        return

    if sel_task == st.session_state.navig['task']:
        return

    # Create task dir
    task_dir = os.path.join(
        st.session_state.paths['out_dir'],
        sel_task
    )
    
    try:
        if not os.path.exists(task_dir):
            os.makedirs(task_dir)
            #st.success(f'Created folder {task_dir}')
            #time.sleep(3)
    except:
        st.error(f'Could not create task folder: {task_dir}')
        return

    # Set task name
    st.session_state.navig['task'] = sel_task
    st.session_state.flags["task"] = True
    st.session_state.paths['task'] = task_dir
    st.session_state.paths['task_curr_path'] = task_dir

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
        
        ####################################
        ### Page settings
        
        # App icon image
        st.session_state.nicon = Image.open("../resources/nichart1.png")

        # Menu navigation
        st.session_state.navig = {
            'main_menu': "Home",
            'workflow': None,
            'pipeline_step': None,
            'task': None
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
            else:
                st.session_state.has_cloud_session = False
        else:
            st.session_state.has_cloud_session = False

        ####################################
        # I/O settings

        # Flags for various i/o
        st.session_state.flags = {
            "out_dir": False,
            "task": False,
            "dicoms": False,
            "dicoms_series": False,
            "nifti": False,
            "T1": False,
            "dlmuse": False,
            "dlmuse_csv": False,
            "dlwmls_csv": False,
            "demog_csv": False,
            "plot_csv": False,
        }

        # Paths to input/output files/folders
        st.session_state.paths = {
            "root": "",
            "init": "",
            "resources": "",
            "proc_def": "",
            "file_search_dir": "",
            "out_dir": "",
            "task": "",
        }

        # Set default directories
        st.session_state.paths["root"] = os.path.dirname(os.path.dirname(os.getcwd()))
        st.session_state.paths["init"] = st.session_state.paths["root"]
        st.session_state.paths["resources"] = os.path.join(
            st.session_state.paths["root"], "resources"
        )
        st.session_state.paths["proc_def"] = os.path.join(
            st.session_state.paths["resources"], "process_definitions"
        )
        if st.session_state.has_cloud_session:
            user_id = st.session_state.cloud_user_id
            st.session_state.paths["out_dir"] = os.path.join(
                st.session_state.paths["root"], "output_folder", user_id
            )
        else:
            st.session_state.paths["out_dir"] = os.path.join(
                st.session_state.paths["root"], "output_folder"
            )
        if not os.path.exists(st.session_state.paths["out_dir"]):
            os.makedirs(st.session_state.paths["out_dir"])
        st.session_state.flags['out_dir'] = True

        # Set default task
        sel_task = 'Experiment_1'
        update_task(sel_task)

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
        update_process_def(st.session_state.paths['proc_def'])

        ####################################
        # Various parameters

        # Average ICV estimated from a large sample
        # IMPORTANT: Used in NiChart Engine for normalization!
        st.session_state.params = {
            'mean_icv': 1430000,
            'harm_min_samples': 30,
        }
        
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





