

import os
from typing import Any

import streamlit as st
import utils.utils_cloud as utilcloud
import utils.utils_dicom as utildcm
import utils.utils_io as utilio
import utils.utils_menu as utilmenu
import utils.utils_nifti as utilni
import utils.utils_session as utilss
import utils.utils_st as utilst
from stqdm import stqdm
import docker
import humanize
import utils.utils_toolloader as tl
import utils.utils_pollstatus as ps
import utils.utils_displayjobs as dj
from pathlib import Path
import pathlib
import time
from streamlit_autorefresh import st_autorefresh

# Page config should be called for each page
utilss.config_page()

utilmenu.menu()

DISABLE_PUBLIC = True
if st.session_state.has_cloud_session and DISABLE_PUBLIC:
    if st.session_state.cloud_user_id != 'b8915a31-544a-4b70-9c7e-253c359e7abe':
        st.header("Page Unavailable")
        st.text("This page is unavailable on NiChart Cloud. Redirecting you to the home page...")
        st.switch_page("pages/home.py")

# Initialize Docker client
try:
    client = docker.from_env()
except:
    st.header("ERROR: We couldn't find Docker!")
    st.markdown("This page is for configuring NiChart to use [Docker](https://www.docker.com) on your system. If you are seeing this message, we couldn't find Docker on your system. Please make sure it is [installed](https://docs.docker.com/engine/install/) or that the Docker daemon is running, then try again.")

st.markdown("This page will refresh automatically every 10 seconds to update the job monitor.")

# Limited to 100 refreshes, in case a user goes AFK.
count = st_autorefresh(interval=10000, limit=100, key='autorefreshcounter')

# Pre-defined list of supported images
SUPPORTED_IMAGES = [
    "cbica/nichart_dlmuse:latest",
]


def get_local_images():
    images = client.images.list()
    local_images = {tag: image for image in images for tag in image.tags}
    return local_images

def pull_image(image_name):
    with st.spinner(f"Pulling {image_name}..."):
        client.images.pull(image_name)
    st.success(f"Pulled {image_name}")

def delete_image(image_name):
    with st.spinner(f"Deleting {image_name}..."):
        client.images.remove(image=image_name, force=True)
    st.success(f"Deleted {image_name}")

def image_disk_usage(image):
    size_bytes = image.attrs['Size']
    return humanize.naturalsize(size_bytes)

# UI starts here

st.title("DEBUG: Try submitting a job to local Docker")
# Button to trigger code execution
path_to_t1 = Path(st.session_state.paths["dir_out"]) / "SampleTool" / "T1"
path_to_outcsv = Path(st.session_state.paths["dir_out"]) / "SampleTool" / "dlmuse.csv"
# Input locations need to already exist. Output locations do not.
path_to_t1.mkdir(parents=True, exist_ok=True)
example_tool_name = "example_tool_template"
example_user_params = {"num_features": 300}
example_user_mounts = {
    "t1_img":  path_to_t1.resolve(),
    "dlmuse_csv": path_to_outcsv.resolve()
}
example_user_mounts = tl.stringify_mounts(example_user_mounts)

# --- Initialize dummy jobs ---
if "active_jobs" not in st.session_state:
    st.session_state.active_jobs = {
        "dummy-job-docker-001": ps.get_handle(mode="dummy", raw_id="placeholder"),
        "dummy-job-aws-002": ps.get_handle(mode="dummy", raw_id="placeholder"),
    }

st.markdown("Job display widget:")
dj.display_jobs()

if st.button("Press me to generate a command!"):
    # Code to run
    # Usage stub
    validated_command = tl.validate_user_request(example_tool_name, example_user_params, example_user_mounts)
    # Display the result
    st.success("Resulting Docker command:")
    st.text(validated_command)
if st.button("Press me to try submitting a local job!"):
    
    result = tl.submit_job(
        tool_name=example_tool_name,
        user_params=example_user_params,
        user_mounts=example_user_mounts,
        execution_mode='local',
    )
    st.rerun()
if st.button("Press me to try submitting a cloud job!"):
    result = tl.submit_job(
        tool_name=example_tool_name,
        user_params=example_user_params,
        user_mounts=example_user_mounts,
        execution_mode='cloud',
    )
    st.text(f"{result}")

if st.button("Press me to try submitting a synchronous cloud job!"):
    progress_bar = stqdm(total=2, desc="Current step", position=0)
    progress_bar.set_description("Submitting job...")
    result = tl.submit_and_run_job_sync(
        tool_name=example_tool_name,
        user_params=example_user_params,
        user_mounts=example_user_mounts,
        execution_mode='cloud',
        progress_bar=progress_bar,
        log=None
    )
    if result.successful:
        st.success(f"Tool {example_tool_name} finished successfully.")
    else:
        st.error(f"Tool {example_tool_name} failed, check error logs.")

indicator_one = st.empty()
indicator_two = st.empty()

def sample_longrunning_task():
    for i in range(5):
        indicator_one.markdown(f"Status: running step {i+1}/5")
        indicator_two.markdown(f"Output: result of step {i+1}")
        time.sleep(1)
    indicator_one.markdown("**Status:** Done ✅")

if st.button("Press me to try automatic UI updates"):
    sample_longrunning_task()

""" st.title("Docker Image Manager")

st.header("Local Docker Images")
local_images = get_local_images()

for image_name in SUPPORTED_IMAGES:
    col1, col2, col3, col4 = st.columns([3, 2, 2, 2])

    with col1:
        st.text(image_name)

    if image_name in local_images:
        with col2:
            st.success("Available Locally")

        with col3:
            size = image_disk_usage(local_images[image_name])
            st.write(f"Size: {size}")

        with col4:
            if st.button(f"Delete {image_name}", key=f"delete_{image_name}"):
                delete_image(image_name)
                st.experimental_rerun()
    else:
        with col2:
            st.warning("Not Available")
        with col3:
            st.write("-")
        with col4:
            if st.button(f"Pull {image_name}", key=f"pull_{image_name}"):
                pull_image(image_name)
                st.experimental_rerun()

st.header("Global Docker Run Options")

st.markdown("Set global options for the `docker run` command:")

if "docker_run_opts" not in st.session_state:
    st.session_state.docker_run_opts = ""

opts = st.text_area("Docker Run Options (e.g., `--rm -it --gpus=all`):", st.session_state.docker_run_opts)

if st.button("Save Docker Options"):
    st.session_state.docker_run_opts = opts
    st.success("Docker options saved")

st.markdown("### Current Global Options:")
st.code(st.session_state.docker_run_opts or "No options set", language="bash")




 """