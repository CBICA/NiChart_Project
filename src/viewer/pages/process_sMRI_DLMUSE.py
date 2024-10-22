import os

import pandas as pd
import streamlit as st
import utils.utils_muse as utilmuse
import utils.utils_nifti as utilni
import utils.utils_st as utilst
import utils.utils_io as utilio

st.markdown(
    """
    - NiChart sMRI segmentation pipeline using DLMUSE.
    - DLMUSE segments raw T1 images into 145 regions of interest (ROIs) + 105 composite ROIs.

    ### Want to learn more?
    - Visit [DLMUSE GitHub](https://github.com/CBICA/NiChart_DLMUSE)
        """
)

# Panel for output (dataset name + out_dir)
utilst.util_panel_workingdir(st.session_state.app_type)

# Panel for selecting input image files
flag_disabled = os.path.exists(st.session_state.paths["dset"]) == False
if st.session_state.app_type == 'CLOUD':
    msg_txt = 'Upload Nifti images'
    utilst.util_upload_folder(st.session_state.paths['T1'], flag_disabled, msg_txt)

else:   # st.session_state.app_type == 'DESKTOP'
    msg_txt = 'Select Nifti images'
    utilst.util_select_folder(
        st.session_state.paths['T1'],
        st.session_state.paths['last_in_dir'],
        flag_disabled,
        msg_txt
    )

# Panel for running DLMUSE
with st.expander(":material/grid_on: Segment image", expanded=False):

    # Device type
    helpmsg = "Choose 'cuda' if your computer has an NVIDIA GPU, 'mps' if you have an Apple M-series chip, and 'cpu' if you have a standard CPU."
    device = utilst.user_input_select(
        "Device", ["cuda", "cpu", "mps"], "dlmuse_sel_device", helpmsg
    )

    # Button to run DLMUSE
    flag_btn = os.path.exists(st.session_state.paths["T1"])
    btn_seg = st.button("Run DLMUSE", disabled=not flag_btn)

    if btn_seg:
        run_dir = os.path.join(st.session_state.paths["root"], "src", "NiChart_DLMUSE")
        if not os.path.exists(st.session_state.paths["DLMUSE"]):
            os.makedirs(st.session_state.paths["DLMUSE"])

        with st.spinner("Wait for it..."):
            dlmuse_cmd = f"NiChart_DLMUSE -i {st.session_state.paths['T1']} -o {st.session_state.paths['DLMUSE']} -d {device}"
            st.info(f"Running: {dlmuse_cmd}", icon=":material/manufacturing:")

            # FIXME : bypass dlmuse run
            os.system(dlmuse_cmd)

            st.success("Run completed!", icon=":material/thumb_up:")

            # Set the dlmuse csv output
            out_csv = f"{st.session_state.paths['DLMUSE']}/DLMUSE_Volumes.csv"
            if os.path.exists(out_csv):
                st.session_state.paths["csv_seg"] = out_csv

# Panel for viewing DLMUSE images
with st.expander(":material/visibility: View segmentations", expanded=False):

    # Set the dlmuse csv output
    st.session_state.paths["csv_seg"] = (
        f"{st.session_state.paths['DLMUSE']}/DLMUSE_Volumes.csv"
    )

    # Selection of MRID
    try:
        df = pd.read_csv(st.session_state.paths["csv_seg"])
        list_mrid = df.MRID.tolist()
    except:
        list_mrid = [""]
    sel_mrid = st.selectbox("MRID", list_mrid, key="selbox_mrid", index=None)

    # Create combo list for selecting target ROI
    list_roi_names = utilmuse.get_roi_names(st.session_state.dicts["muse_sel"])
    sel_var = st.selectbox("ROI", list_roi_names, key="selbox_rois", index=0)

    # Detect list of ROI indices to display
    list_sel_rois = utilmuse.get_derived_rois(
        sel_var,
        st.session_state.dicts["muse_derived"]
    )

    # Create a list of checkbox options
    list_orient = st.multiselect("Select viewing planes:", utilni.VIEWS, utilni.VIEWS)

    # View hide overlay
    is_show_overlay = st.checkbox("Show overlay", True)

    # Crop to mask area
    crop_to_mask = st.checkbox("Crop to mask", True)

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
            for i, tmp_orient in enumerate(list_orient):
                with blocks[i]:
                    ind_view = utilni.VIEWS.index(tmp_orient)
                    if is_show_overlay is False:
                        utilst.show_img3D(
                            img, ind_view, mask_bounds[ind_view, :], tmp_orient
                        )
                    else:
                        utilst.show_img3D(
                            img_masked, ind_view, mask_bounds[ind_view, :], tmp_orient
                        )

# Panel for downloading results
if st.session_state.app_type == "CLOUD":
    with st.expander(":material/download: Download Results", expanded=False):

        # Zip results and download
        flag_btn = os.path.exists(st.session_state.paths[st.session_state.sel_mod])
        out_zip = bytes()
        if flag_btn:
            if not os.path.exists(st.session_state.paths["OutZipped"]):
                os.makedirs(st.session_state.paths["OutZipped"])
            f_tmp = os.path.join(st.session_state.paths["OutZipped"], "DLMUSE.zip")
            out_zip = utilio.zip_folder(st.session_state.paths["DLMUSE"], f_tmp)

        st.download_button(
            "Download DLMUSE results",
            out_zip,
            file_name=f"{st.session_state.sel_mod}.zip",
            disabled=not flag_btn,
        )

with st.expander("TMP: session vars"):
    st.write(st.session_state)
with st.expander("TMP: session vars - paths"):
    st.write(st.session_state.paths)
