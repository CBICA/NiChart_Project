import logging
import os

import NiChart_DLMUSE as ncd
import pandas as pd
import streamlit as st
import utils.utils_cloud as utilcloud
import utils.utils_io as utilio
import utils.utils_nifti as utilni
import utils.utils_pages as utilpg
import utils.utils_panels as utilpn
import utils.utils_rois as utilroi
import utils.utils_st as utilst
from stqdm import stqdm


def panel_dlmuse() -> None:
    """
    Panel for running dlmuse
    """
    with st.container(border=True):
        # Device type
        if st.session_state.app_type != "cloud":
            helpmsg = "Choose 'cuda' if your computer has an NVIDIA GPU, 'mps' if you have an Apple M-series chip, and 'cpu' if you have a standard CPU."
            device = utilst.user_input_select(
                "Device",
                "key_select_device",
                ["cuda", "cpu", "mps"],
                None,
                helpmsg,
                False,
            )
        else:
            device = "cuda"

        # Button to run DLMUSE
        btn_seg = st.button("Run DLMUSE", disabled=False)
        if btn_seg:
            if not os.path.exists(st.session_state.paths["dlmuse"]):
                os.makedirs(st.session_state.paths["dlmuse"])

            with st.spinner("Wait for it..."):
                fcount = utilio.get_file_count(st.session_state.paths["T1"])
                if st.session_state.has_cloud_session:
                    utilcloud.update_stats_db(
                        st.session_state.cloud_user_id, "DLMUSE", fcount
                    )

                progress_bar = stqdm(total=9, desc="Current step", position=0)
                progress_bar.set_description("Starting...")

                # Clear logs to avoid huge accumulation
                # TODO: This is hacky, fix this more elegantly from NiChart_DLMUSE by using log rotation
                if os.path.exists("pipeline.log"):
                    try:
                        open("pipeline.log", "w").close()
                    except:
                        print("Could not empty pipeline.log!")

                ncd.run_pipeline(
                    st.session_state.paths["T1"],
                    st.session_state.paths["dlmuse"],
                    device,
                    dlmuse_extra_args="-nps 1 -npp 1",
                    dlicv_extra_args="-nps 1 -npp 1",
                    progress_bar=progress_bar,
                )
                # Reset logging level after NiChart_DLMUSE pipeline changes it...
                # TODO: Fix this hack just like the above
                logging.basicConfig(
                    filename="pipeline.log", encoding="utf-8", level=logging.ERROR
                )

                # dlmuse_cmd = f"NiChart_DLMUSE -i {st.session_state.paths['T1']} -o {st.session_state.paths['dlmuse']} -d {device} --cores 1"
                # st.info(f"Running: {dlmuse_cmd}", icon=":material/manufacturing:")

                # FIXME : bypass dlmuse run
                # print(f"About to run: {dlmuse_cmd}")
                # os.system(dlmuse_cmd)

        out_csv = f"{st.session_state.paths['dlmuse']}/DLMUSE_Volumes.csv"
        num_dlmuse = utilio.get_file_count(st.session_state.paths["dlmuse"], ".nii.gz")
        if os.path.exists(out_csv):
            st.session_state.paths["dlmuse_csv"] = out_csv
            st.session_state.flags["dlmuse_csv"] = True
            st.success(
                f"DLMUSE images are ready ({st.session_state.paths['dlmuse']}, {num_dlmuse} scan(s))",
                icon=":material/thumb_up:",
            )

            with st.expander("View DLMUSE volumes"):
                df_dlmuse = pd.read_csv(st.session_state.paths["dlmuse_csv"])
                st.dataframe(df_dlmuse)

        s_title = "DLMUSE Segmentation"
        s_text = """
        - Raw T1 images are segmented into anatomical regions of interest (ROIs) using DLMUSE.
        - The output folder (**"DLMUSE"**) will contain the segmentation mask for each scan, and a single CSV file with volumes of all ROIs. The result file will include single ROIs (segmented regions) and composite ROIs (obtained by merging single ROIs within a tree structure).
        """
        utilst.util_help_dialog(s_title, s_text)


def panel_view() -> None:
    """
    Panel for viewing images
    """
    with st.container(border=True):
        # Selection of MRID
        try:
            df = pd.read_csv(st.session_state.paths["dlmuse_csv"])
            list_mrid = df.MRID.tolist()
        except:
            list_mrid = []
        if len(list_mrid) == 0:
            st.warning("Result file is empty!")
            return

        sel_mrid = st.selectbox(
            "MRID", list_mrid, key="selbox_mrid", index=None, disabled=False
        )
        if sel_mrid is None:
            st.warning("Please select the MRID!")
            return

        # Create combo list for selecting target ROI
        list_roi_names = utilroi.get_roi_names(st.session_state.dicts["muse_sel"])
        sel_var = st.selectbox(
            "ROI", list_roi_names, key="selbox_rois", index=None, disabled=False
        )
        if sel_var is None:
            st.warning("Please select the ROI!")
            return

        # Create a list of checkbox options
        list_orient = st.multiselect(
            "Select viewing planes:", utilni.img_views, utilni.img_views, disabled=False
        )

        if list_orient is None or len(list_orient) == 0:
            st.warning("Please select the viewing plane!")
            return

        # View hide overlay
        is_show_overlay = st.checkbox("Show overlay", True, disabled=False)

        # Crop to mask area
        crop_to_mask = st.checkbox("Crop to mask", True, disabled=False)

        # Get indices for the selected var
        list_rois = utilroi.get_list_rois(
            sel_var,
            st.session_state.rois["roi_dict_inv"],
            st.session_state.rois["roi_dict_derived"],
        )

        if list_rois is None:
            st.warning("ROI list is empty!")
            return

        # Select images
        if sel_mrid is None:
            st.warning("Please select the MRID!")
            return

        st.session_state.paths["sel_img"] = utilio.get_image_path(
            st.session_state.paths["T1"], sel_mrid, ["nii.gz", ".nii"]
        )

        st.session_state.paths["sel_seg"] = utilio.get_image_path(
            st.session_state.paths["dlmuse"], sel_mrid, ["nii.gz", ".nii"]
        )

        if not os.path.exists(st.session_state.paths["sel_img"]):
            st.warning("Could not locate underlay image!")
            return

        if not os.path.exists(st.session_state.paths["sel_seg"]):
            st.warning("Could not locate overlay image!")
            return

        with st.spinner("Wait for it..."):
            # Process image and mask to prepare final 3d matrix to display
            img, mask, img_masked = utilni.prep_image_and_olay(
                st.session_state.paths["sel_img"],
                st.session_state.paths["sel_seg"],
                list_rois,
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
                    size_auto = True
                    if is_show_overlay is False:
                        utilst.show_img3D(
                            img,
                            ind_view,
                            mask_bounds[ind_view, :],
                            tmp_orient,
                            size_auto,
                        )
                    else:
                        utilst.show_img3D(
                            img_masked,
                            ind_view,
                            mask_bounds[ind_view, :],
                            tmp_orient,
                            size_auto,
                        )


# Page config should be called for each page
utilpg.config_page()
utilpg.select_main_menu()
utilpg.select_pipeline()
utilpg.select_pipeline_step()

with st.container(border=True):

    st.markdown(
        """
        ### Segmentation of Anatomical Regions of Interest
        - Segmentation of T1-weighted MRI scans into anatomical regions of interest (ROIs)
        - [DLMUSE](https://github.com/CBICA/NiChart_DLMUSE): Fast deep learning based segmentation into 145 ROIs + 105 composite ROIs
        """
    )

    st.markdown("##### Select Task")
    list_tasks = ["Experiment Name", "Input Data", "DLMUSE", "View Scans", "Download"]
    sel_task = st.pills(
        "Select Task", list_tasks, selection_mode="single", label_visibility="collapsed"
    )
    if sel_task == "Experiment Name":
        utilpn.util_panel_experiment()
    elif sel_task == "Input Data":
        status = st.session_state.flags["experiment"]
        utilpn.util_panel_input_multi("T1", status)
    elif sel_task == "DLMUSE":
        panel_dlmuse()
    elif sel_task == "View Scans":
        panel_view()
    elif sel_task == "Download":
        utilpn.util_panel_download("DLMUSE")
