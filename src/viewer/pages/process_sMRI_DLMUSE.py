import os

import pandas as pd
import streamlit as st
import utils.utils_muse as utilmuse
import utils.utils_nifti as utilni
import utils.utils_st as utilst

st.markdown(
    """
    - NiChart sMRI segmentation pipeline using DLMUSE.
    - DLMUSE segments raw T1 images into 145 regions of interest (ROIs) + 105 composite ROIs.

    ### Want to learn more?
    - Visit [DLMUSE GitHub](https://github.com/CBICA/NiChart_DLMUSE)
        """
)

# Panel for output (dataset name + out_dir)
utilst.util_panel_workingdir()

# Panel for running DLMUSE
with st.expander("Run DLMUSE", expanded=False):

    # Input T1 image folder
    helpmsg = "DLMUSE will be applied to .nii/.nii.gz images directly in the input folder.\n\nChoose the path by typing it into the text field or using the file browser to browse and select it"
    st.session_state.paths['T1'] = utilst.user_input_folder(
        "Select folder",
        "btn_indir_t1",
        "Input folder",
        st.session_state.paths["last_sel"],
        st.session_state.paths['T1'],
        helpmsg,
    )

    # Device type
    helpmsg = "Choose 'cuda' if your computer has an NVIDIA GPU, 'mps' if you have an Apple M-series chip, and 'cpu' if you have a standard CPU."
    device = utilst.user_input_select(
        "Device", ["cuda", "cpu", "mps"], "dlmuse_sel_device", helpmsg
    )

    # Button to run DLMUSE
    flag_btn = os.path.exists(st.session_state.paths['T1'])
    btn_dlmuse = st.button("Run DLMUSE", disabled=not flag_btn)

    if btn_dlmuse:
        run_dir = os.path.join(
            st.session_state.paths["root"], "src", "NiChart_DLMUSE"
        )
        if not os.path.exists(st.session_state.paths["DLMUSE"]):
            os.makedirs(st.session_state.paths["DLMUSE"])

        with st.spinner("Wait for it..."):
            dlmuse_cmd = f"NiChart_DLMUSE -i {st.session_state.paths['T1']} -o {st.session_state.paths["DLMUSE"]} -d {device}"
            st.info(f"Running: {dlmuse_cmd}", icon=":material/manufacturing:")

            # FIXME : bypass dlmuse run
            os.system(dlmuse_cmd)

            st.success("Run completed!", icon=":material/thumb_up:")

            # Set the dlmuse csv output
            out_csv = f"{st.session_state.paths['DLMUSE']}/DLMUSE_Volumes.csv"
            if os.path.exists(out_csv):
                st.session_state.paths["csv_dlmuse"] = out_csv

# Panel for viewing DLMUSE images
with st.expander("View segmentations", expanded=False):

    # Set the dlmuse csv output
    st.session_state.paths["csv_dlmuse"] = f"{st.session_state.paths["DLMUSE"]}/DLMUSE_Volumes.csv"

    # Selection of MRID
    try:
        df = pd.read_csv(st.session_state.paths["csv_dlmuse"])
        list_mrid = df.MRID.tolist()
    except:
        list_mrid = ['']
    sel_mrid = st.selectbox("MRID", list_mrid, key="selbox_mrid", index=None)

    # Select ROI
    dict_roi, dict_derived = utilmuse.read_derived_roi_list(
        st.session_state.dicts["muse_sel"], st.session_state.dicts["muse_derived"]
    )
    sel_var = st.selectbox("ROI", list(dict_roi.keys()), key="selbox_rois", index=0)
    sel_var_ind = dict_roi[sel_var]

    # Create a list of checkbox options
    list_orient = st.multiselect(
        "Select viewing planes:", utilni.VIEWS, utilni.VIEWS
    )

    # View hide overlay
    is_show_overlay = st.checkbox("Show overlay", True)

    # Select images
    flag_img = False
    if sel_mrid is not None:
        st.session_state.paths["sel_img"] = os.path.join(
            st.session_state.paths['T1'], sel_mrid + st.session_state.suff_t1img
        )
        st.session_state.paths["sel_dlmuse"] = os.path.join(
            st.session_state.paths["DLMUSE"], sel_mrid + st.session_state.suff_dlmuse
        )

        flag_img = os.path.exists(st.session_state.paths["sel_img"]) and os.path.exists(st.session_state.paths["sel_dlmuse"])

    if flag_img:

        with st.spinner("Wait for it..."):

            # Process image and mask to prepare final 3d matrix to display
            img, mask, img_masked = utilni.prep_image_and_olay(
                st.session_state.paths["sel_img"],
                st.session_state.paths["sel_dlmuse"],
                sel_var_ind,
                dict_derived,
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

with st.expander('TMP: session vars'):
    st.write(st.session_state)
with st.expander('TMP: session vars - paths'):
    st.write(st.session_state.paths)
