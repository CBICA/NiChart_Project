import os

import numpy as np
import pandas as pd
import streamlit as st
import utils.utils_muse as utilmuse
import utils.utils_nifti as utilni
import utils.utils_st as utilst
from stqdm import stqdm

st.markdown(
    """
    - View image (underlay) and segmentation (overlay)
    """
)

# Panel for output (dataset name + out_dir)
utilst.util_panel_workingdir(st.session_state.app_type)

# Set default path for the data csv
if os.path.exists(st.session_state.paths["csv_mlscores"]):
    st.session_state.paths["csv_plot"] = st.session_state.paths["csv_mlscores"]
elif os.path.exists(st.session_state.paths["csv_dlmuse"]):
    st.session_state.paths["csv_plot"] = st.session_state.paths["csv_dlmuse"]

# Panel for selecting input csv
flag_disabled = not st.session_state.flags['dset']

if st.session_state.app_type == "CLOUD":
    with st.expander(":material/upload: Upload data", expanded=False):  # type:ignore
        utilst.util_upload_file(
            st.session_state.paths["csv_dlmuse"],
            "Input data csv file",
            "key_in_csv",
            flag_disabled,
            "visible",
        )
        if not flag_disabled and os.path.exists(st.session_state.paths["csv_plot"]):
            st.success(f"Data is ready ({st.session_state.paths["csv_plot"]})", icon=":material/thumb_up:")

else:  # st.session_state.app_type == 'DESKTOP'
    with st.expander(":material/upload: Select data", expanded=False):
        utilst.util_select_file(
            "selected_data_file",
            "Data csv",
            st.session_state.paths["csv_dlmuse"],
            st.session_state.paths["last_in_dir"],
            flag_disabled,
        )
        if not flag_disabled and os.path.exists(st.session_state.paths["csv_dlmuse"]):
            st.success(f"Data is ready ({st.session_state.paths["csv_dlmuse"]})", icon=":material/thumb_up:")

    with st.expander(":material/upload: Select data", expanded=False):  # type:ignore
        utilst.util_select_folder(
            "key_sel_img_folder_viewer",
            "T1 nifti image folder",
            st.session_state.paths["T1"],
            st.session_state.paths["last_in_dir"],
            flag_disabled,
        )
        if not flag_disabled:
            fcount = utilio.get_file_count(st.session_state.paths["T1"])
            if fcount > 0:
                st.session_state.flags['T1'] = True
                st.success(
                    f"Data is ready ({st.session_state.paths["T1"]}, {fcount} files)",
                    icon=":material/thumb_up:"
                )

    with st.expander(":material/upload: Select data", expanded=False):  # type:ignore
        utilst.util_select_folder(
            "key_sel_dlmuse_folder_viewer",
            "T1 nifti image folder",
            st.session_state.paths["DLMUSE"],
            st.session_state.paths["last_in_dir"],
            flag_disabled,
        )
        if not flag_disabled:
            fcount = utilio.get_file_count(st.session_state.paths["DLMUSE"])
            if fcount > 0:
                st.session_state.flags['DLMUSE'] = True
                st.success(
                    f"Data is ready ({st.session_state.paths["DLMUSE"]}, {fcount} files)",
                    icon=":material/thumb_up:"
                )

    # T1 suffix
    suff_t1img = utilst.user_input_text(
        "T1 img suffix", st.session_state.suff_t1img, 'T1 img suffix', flag_disabled
    )
    st.session_state.suff_t1img = suff_t1img

    # DLMUSE suffix
    suff_seg = utilst.user_input_text(
        "DLMUSE image suffix", st.session_state.suff_seg, 'DLMUSE img suffix', flag_disabled
    )
    st.session_state.suff_seg = suff_seg

# Panel for viewing DLMUSE images
with st.expander(":material/visibility: View segmentations", expanded=False):

    flag_disabled = not st.session_state.flags['csv_dlmuse']

    # Selection of MRID
    try:
        df = pd.read_csv(st.session_state.paths["csv_dlmuse"])
        list_mrid = df.MRID.tolist()
    except:
        list_mrid = [""]
    sel_mrid = st.selectbox("MRID", list_mrid, key="selbox_mrid", index=None, disabled = flag_disabled)

    # Create combo list for selecting target ROI
    list_roi_names = utilmuse.get_roi_names(st.session_state.dicts["muse_sel"])
    sel_var = st.selectbox("ROI", list_roi_names, key="selbox_rois", index=None, disabled = flag_disabled)

    # Create a list of checkbox options
    list_orient = st.multiselect(
        "Select viewing planes:",
        utilni.img_views,
        utilni.img_views,
        disabled = flag_disabled
    )

    # View hide overlay
    is_show_overlay = st.checkbox("Show overlay", True, disabled = flag_disabled)

    # Crop to mask area
    crop_to_mask = st.checkbox("Crop to mask", True, disabled = flag_disabled)

    if not flag_disabled:

        # Detect list of ROI indices to display
        list_sel_rois = utilmuse.get_derived_rois(
            sel_var, st.session_state.dicts["muse_derived"]
        )

        # Select images
        flag_img = False
        if sel_mrid is not None:
            st.session_state.paths["sel_img"] = os.path.join(
                st.session_state.paths["T1"], sel_mrid + st.session_state.suff_t1img
            )
            st.session_state.paths["sel_seg"] = os.path.join(
                st.session_state.paths["DLMUSE"], sel_mrid + st.session_state.suff_seg
            )

            flag_img = os.path.exists(st.session_state.paths["sel_img"]) and os.path.exists(
                st.session_state.paths["sel_seg"]
            )

        if flag_img:
            with st.spinner("Wait for it..."):

                # Process image and mask to prepare final 3d matrix to display
                img, mask, img_masked = utilni.prep_image_and_olay(
                    st.session_state.paths["sel_img"],
                    st.session_state.paths["sel_seg"],
                    list_sel_rois,
                    crop_to_mask,
                )

                # Detect mask bounds and center in each view
                mask_bounds = utilni.detect_mask_bounds(mask)

                # Show images
                blocks = st.columns(len(list_orient))
                for i, tmp_orient in stqdm(
                    enumerate(list_orient),
                    desc="Showing images ...",
                    total=len(list_orient),
                ):
                    with blocks[i]:
                        ind_view = utilni.img_views.index(tmp_orient)
                        if is_show_overlay is False:
                            utilst.show_img3D(
                                img, ind_view, mask_bounds[ind_view, :], tmp_orient
                            )
                        else:
                            utilst.show_img3D(
                                img_masked, ind_view, mask_bounds[ind_view, :], tmp_orient
                            )
