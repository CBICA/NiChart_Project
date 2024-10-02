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
flag_expanded = st.session_state.paths["dset"] == ""
with st.expander("Select output", expanded=flag_expanded):
    # Dataset name: All results will be saved in a main folder named by the dataset name
    helpmsg = "Each dataset's results are organized in a dedicated folder named after the dataset"
    dset_name = utilst.user_input_text(
        "Dataset name", st.session_state.dset_name, helpmsg
    )

    # Out folder
    helpmsg = "DLMUSE images will be saved to the output folder.\n\nChoose the path by typing it into the text field or using the file browser to browse and select it"
    path_out = utilst.user_input_folder(
        "Select folder",
        "btn_sel_out_dir",
        "Output folder",
        st.session_state.paths["last_sel"],
        st.session_state.paths["out"],
        helpmsg,
    )
    if dset_name != "" and path_out != "":
        st.session_state.dset_name = dset_name
        st.session_state.paths["out"] = path_out
        st.session_state.paths["dset"] = os.path.join(path_out, dset_name)
        st.session_state.paths["dlmuse"] = os.path.join(path_out, dset_name, "DLMUSE")
        st.success(f'Results will be saved to: {st.session_state.paths['dlmuse']}')

# Panel for running DLMUSE
if st.session_state.dset_name != "":
    with st.expander("Run DLMUSE", expanded=True):

        # Input T1 image folder
        helpmsg = "DLMUSE will be applied to .nii/.nii.gz images directly in the input folder.\n\nChoose the path by typing it into the text field or using the file browser to browse and select it"
        path_t1 = utilst.user_input_folder(
            "Select folder",
            "btn_indir_t1",
            "Input folder",
            st.session_state.paths["last_sel"],
            st.session_state.paths["t1"],
            helpmsg,
        )
        st.session_state.paths["t1"] = path_t1

        # Device type
        helpmsg = "Choose 'cuda' if your computer has an NVIDIA GPU, 'mps' if you have an Apple M-series chip, and 'cpu' if you have a standard CPU."
        device = utilst.user_input_select(
            "Device", ["cuda", "cpu", "mps"], "dlmuse_sel_device", helpmsg
        )

        # Button to run DLMUSE
        flag_btn = os.path.exists(st.session_state.paths["t1"])
        btn_dlmuse = st.button("Run DLMUSE", disabled=not flag_btn)

        if btn_dlmuse:
            run_dir = os.path.join(
                st.session_state.paths["root"], "src", "NiChart_DLMUSE"
            )
            if not os.path.exists(st.session_state.paths["dlmuse"]):
                os.makedirs(st.session_state.paths["dlmuse"])

            with st.spinner("Wait for it..."):
                dlmuse_cmd = f"NiChart_DLMUSE -i {st.session_state.paths['t1']} -o {st.session_state.paths['dlmuse']} -d {device}"
                st.info(f"Running: {dlmuse_cmd}", icon=":material/manufacturing:")
                os.system(dlmuse_cmd)
                st.success("Run completed!", icon=":material/thumb_up:")

                # Set the dlmuse csv output
                out_csv = f"{st.session_state.paths['dlmuse']}/DLMUSE_Volumes.csv"
                if os.path.exists(out_csv):
                    st.session_state.paths["csv_dlmuse"] = out_csv

# Panel for viewing DLMUSE images
if os.path.exists(st.session_state.paths["csv_dlmuse"]):
    with st.expander("View segmentations", expanded=True):

        # Read dlmuse csv
        df = pd.read_csv(st.session_state.paths["csv_dlmuse"])

        # Create a dictionary of MUSE indices and names
        df_muse = pd.read_csv(st.session_state.dicts["muse_all"])

        # Read derived roi list and convert to a dict
        dict_roi, dict_derived = utilmuse.read_derived_roi_list(
            st.session_state.dicts["muse_sel"], st.session_state.dicts["muse_derived"]
        )

        # Selection of MRID
        sel_mrid = st.selectbox("MRID", df.MRID.tolist(), key="selbox_mrid", index=0)

        st.session_state.paths["sel_img"] = os.path.join(
            st.session_state.paths["t1"], sel_mrid + st.session_state.suff_t1img
        )
        st.session_state.paths["sel_dlmuse"] = os.path.join(
            st.session_state.paths["dlmuse"], sel_mrid + st.session_state.suff_dlmuse
        )

        # Selection of ROI
        sel_var = st.selectbox("ROI", list(dict_roi.keys()), key="selbox_rois", index=0)
        sel_var_ind = dict_roi[sel_var]

        # Create a list of checkbox options
        list_orient = st.multiselect(
            "Select viewing planes:", utilni.VIEWS, utilni.VIEWS
        )

        # View hide overlay
        is_show_overlay = st.checkbox("Show overlay", True)

        flag_btn = os.path.exists(st.session_state.paths["sel_img"])

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


# FIXME: this is for debugging; will be removed
with st.expander("session_state: All"):
    st.write(st.session_state)
