import os

import numpy as np
import pandas as pd
import streamlit as st
import utils.utils_muse as utilmuse
import utils.utils_nifti as utilni
import utils.utils_st as utilst

# Parameters for viewer
VIEWS = ["axial", "coronal", "sagittal"]
VIEW_AXES = [0, 2, 1]
VIEW_OTHER_AXES = [(1, 2), (0, 1), (0, 2)]
MASK_COLOR = (0, 255, 0)  # RGB format
MASK_COLOR = np.array([0.0, 1.0, 0.0])  # RGB format
OLAY_ALPHA = 0.2


# Page controls in side bar
# with st.sidebar:

f_img = ""
f_mask = ""

# Selection of subject list and image paths
with st.expander("Select subject list, image paths and suffixes"):

    # DLMUSE file name
    helpmsg = "Input csv file with DLMUSE ROI volumes.\n\nUsed for selecting the MRID and the ROI name.\n\nChoose the file by typing it into the text field or using the file browser to browse and select it"
    csv_dlmuse, csv_path = utilst.user_input_file(
        "Select file",
        "btn_input_dlmuse",
        "Subject list",
        st.session_state.paths["last_sel"],
        st.session_state.paths["csv_dlmuse"],
        helpmsg,
    )
    if os.path.exists(csv_dlmuse):
        st.session_state.paths["csv_dlmuse"] = csv_dlmuse
        st.session_state.paths["last_sel"] = csv_path

    # Input T1 image folder
    helpmsg = "Path to T1 images.\n\nChoose the path by typing it into the text field or using the file browser to browse and select it"
    path_t1 = utilst.user_input_folder(
        "Select folder",
        "btn_indir_t1",
        "T1 folder",
        st.session_state.paths["last_sel"],
        st.session_state.paths["T1"],
        helpmsg,
    )
    st.session_state.paths["T1"] = path_t1

    # Input DLMUSE image folder
    helpmsg = "Path to DLMUSE images.\n\nChoose the path by typing it into the text field or using the file browser to browse and select it"
    path_dlmuse = utilst.user_input_folder(
        "Select folder",
        "btn_indir_dlmuse",
        "DLMUSE folder",
        st.session_state.paths["last_sel"],
        st.session_state.paths["dlmuse"],
        helpmsg,
    )
    st.session_state.paths["dlmuse"] = path_dlmuse

    # T1 suffix
    suff_t1img = utilst.user_input_text(
        "T1 img suffix", st.session_state.suff_t1img, helpmsg
    )
    st.session_state.suff_t1img = suff_t1img

    # DLMUSE suffix
    suff_dlmuse = utilst.user_input_text(
        "DLMUSE image suffix", st.session_state.suff_dlmuse, helpmsg
    )
    st.session_state.suff_dlmuse = suff_dlmuse


# Selection of MRID and ROI name
if os.path.exists(st.session_state.paths["csv_dlmuse"]):

    with st.container(border=True):

        df = pd.read_csv(st.session_state.paths["csv_dlmuse"])

        # Create a dictionary of MUSE indices and names
        df_muse = pd.read_csv(st.session_state.dicts["muse_all"])

        # df_muse = df_muse[df_muse.Name.isin(df.columns)]
        # dict_roi = dict(zip(df_muse.Name, df_muse.Index))

        # Read derived roi list and convert to a dict
        dict_roi, dict_derived = utilmuse.read_derived_roi_list(
            st.session_state.dicts["muse_sel"], st.session_state.dicts["muse_derived"]
        )

        # Selection of MRID
        sel_mrid = st.session_state.sel_mrid
        if sel_mrid == "":
            sel_ind = 0
            sel_type = "(auto)"
        else:
            sel_ind = df.MRID.tolist().index(sel_mrid)
            sel_type = "(user)"
        sel_mrid = st.selectbox(
            "MRID", df.MRID.tolist(), key="selbox_mrid", index=sel_ind
        )

        # Selection of ROI
        #  - The variable will be selected from the active plot

        sel_var = ""
        try:
            sel_var = st.session_state.plots.loc[st.session_state.plot_active, "yvar"]
        except:
            print("Could not detect an active plot")
        if sel_var == "":
            sel_ind = 2
            sel_var = list(dict_roi.keys())[0]
            sel_type = "(auto)"
        else:
            sel_ind = df_muse.Name.tolist().index(sel_var)
            sel_type = "(user)"
        sel_var = st.selectbox(
            "ROI", list(dict_roi.keys()), key="selbox_rois", index=sel_ind
        )

    with st.container(border=True):

        # Create a list of checkbox options
        # list_orient = st.multiselect("Select viewing planes:", VIEWS, VIEWS[0])
        list_orient = st.multiselect("Select viewing planes:", VIEWS, VIEWS)

        # View hide overlay
        is_show_overlay = st.checkbox("Show overlay", True)

    # Select roi index
    sel_var_ind = dict_roi[sel_var]

    # File names for img and mask
    f_img = os.path.join(
        st.session_state.paths["out"],
        st.session_state.paths["T1"],
        sel_mrid + st.session_state.suff_t1img,
    )

    f_mask = os.path.join(
        st.session_state.paths["out"],
        st.session_state.paths["dlmuse"],
        sel_mrid + st.session_state.suff_dlmuse,
    )

if os.path.exists(f_img) & os.path.exists(f_mask):

    # Process image and mask to prepare final 3d matrix to display
    img, mask, img_masked = utilni.prep_image_and_olay(
        f_img, f_mask, sel_var_ind, dict_derived
    )

    # Detect mask bounds and center in each view
    mask_bounds = utilni.detect_mask_bounds(mask)

    # Show images
    blocks = st.columns(len(list_orient))
    for i, tmp_orient in enumerate(list_orient):
        with blocks[i]:
            ind_view = VIEWS.index(tmp_orient)
            if is_show_overlay is False:
                utilst.show_img3D(img, ind_view, mask_bounds[ind_view, :], tmp_orient)
            else:
                utilst.show_img3D(
                    img_masked, ind_view, mask_bounds[ind_view, :], tmp_orient
                )

else:
    if not os.path.exists(f_img):
        st.warning(f"Image not found: {f_img}")
    else:
        st.warning(f"Mask not found: {f_mask}")
