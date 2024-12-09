import os

import pandas as pd
import streamlit as st
import utils.utils_io as utilio
import utils.utils_menu as utilmenu
import utils.utils_nifti as utilni
import utils.utils_session as utilss
import utils.utils_st as utilst
from stqdm import stqdm

# Page config should be called for each page
utilss.config_page()

utilmenu.menu()

st.write("# Segmentation of White Matter Lesions")

st.markdown(
    """
    - Segmentation of WM Lesions on FL scan
    - [DLWMLS](https://github.com/CBICA/NiChart_DLWMLS): Fast deep learning based segmentation of WM lesions
    """
)


def panel_wdir() -> None:
    """
    Panel for selecting the working dir
    """
    icon = st.session_state.icon_thumb[st.session_state.flags["dir_out"]]
    show_panel_wdir = st.checkbox(
        f":material/folder_shared: Working Directory {icon}", value=False
    )
    if not show_panel_wdir:
        return

    with st.container(border=True):
        utilst.util_panel_workingdir(st.session_state.app_type)
        if os.path.exists(st.session_state.paths["dset"]):
            st.success(
                f"All results will be saved to: {st.session_state.paths['dset']}",
                icon=":material/thumb_up:",
            )
            st.session_state.flags["dir_out"] = True

        utilst.util_workingdir_get_help()

def panel_infl() -> None:
    """
    Panel for uploading input t1 images
    """
    st.session_state.flags["dir_fl"] = os.path.exists(st.session_state.paths["FL"])

    msg = st.session_state.app_config[st.session_state.app_type]["msg_infile"]
    icon = st.session_state.icon_thumb[st.session_state.flags["dir_fl"]]
    show_panel_infl = st.checkbox(
        f":material/upload: {msg} FL Images {icon}",
        disabled=not st.session_state.flags["dir_out"],
        value=False,
    )
    if not show_panel_infl:
        return

    with st.container(border=True):
        if st.session_state.app_type == "cloud":
            utilst.util_upload_folder(
                st.session_state.paths["FL"],
                "FL images",
                False,
                "Nifti images can be uploaded as a folder, multiple files, or a single zip file",
            )
            fcount = utilio.get_file_count(st.session_state.paths["FL"])
            if fcount > 0:
                st.session_state.flags["FL"] = True
                path_fl = st.session_state.paths["FL"]
                st.success(
                    f"Data is ready ({path_fl}, {fcount} files)",
                    icon=":material/thumb_up:",
                )

        else:  # st.session_state.app_type == 'desktop'
            utilst.util_select_folder(
                "selected_img_folder",
                "FL nifti image folder",
                st.session_state.paths["FL"],
                st.session_state.paths["file_search_dir"],
                False,
            )
            fcount = utilio.get_file_count(st.session_state.paths["FL"])
            if fcount > 0:
                st.session_state.flags["dir_fl"] = True
                path_fl = st.session_state.paths["FL"]
                st.success(
                    f"Data is ready ({path_fl}, {fcount} files)",
                    icon=":material/thumb_up:",
                )

        s_title="Input FL Scans"
        s_text="""
        - Upload or select input FL scans. DLWMLS can be directly applied to raw FL scans. Nested folders are not supported.

        - The result file with total segmented WMLS volume includes an **"MRID"** column that uniquely identifies each scan. **MRID** is extracted from image file names by removing the common suffix to all images. Using consistent input image names is **strongly recommended**

        - On the desktop app, a symbolic link named **"Nifti/FL"** will be created in the **working directory**, pointing to your input FL images folder.

        - On the cloud platform, you can directly drag and drop your FL image files or folder and they will be uploaded to the **"Nifti/FL"** folder within the **working directory**.

        - On the cloud, **we strongly recommend** compressing your input images into a single ZIP archive before uploading. The system will automatically extract the contents of the ZIP file into the **"Nifti/T1"** folder upon upload.
        """
        utilst.util_get_help(s_title, s_text)

def panel_dlwmls() -> None:
    """
    Panel for running DLWMLS
    """
    icon = st.session_state.icon_thumb[st.session_state.flags["csv_dlwmls"]]
    show_panel_dlwmls = st.checkbox(
        f":material/new_label: Run DLWMLS {icon}",
        disabled=not st.session_state.flags["dir_fl"],
        value=False,
    )
    if not show_panel_dlwmls:
        return

    with st.container(border=True):
        # Device type
        helpmsg = "Choose 'cuda' if your computer has an NVIDIA GPU, 'mps' if you have an Apple M-series chip, and 'cpu' if you have a standard CPU."
        if st.session_state.app_type != 'cloud':
            device = utilst.user_input_select(
                "Device", "key_select_device", ["cuda", "cpu", "mps"], None, helpmsg, False
            )
        else:
            device = 'cuda'

        # Button to run DLWMLS
        btn_seg = st.button("Run DLWMLS", disabled=False)
        if btn_seg:
            if not os.path.exists(st.session_state.paths["dlwmls"]):
                os.makedirs(st.session_state.paths["dlwmls"])

            with st.spinner("Wait for it..."):
                dlwmls_cmd = f"DLWMLS -i {st.session_state.paths['FL']} -o {st.session_state.paths['dlwmls']} -d {device}"
                st.info(f"Running: {dlwmls_cmd}", icon=":material/manufacturing:")

                # FIXME : bypass dlwmls run
                print(f"About to run: {dlwmls_cmd}")
                os.system(dlwmls_cmd)

                run_scr = os.path.join(
                    st.session_state.paths["root"],
                    "src",
                    "workflows",
                    "w_DLWMLS",
                    "wmls_post.py",
                )
                post_dlwmls_cmd = f"python {run_scr} --in_dir {st.session_state.paths['dlwmls']} --in_suff _FL_WMLS.nii.gz --out_csv {os.path.join(st.session_state.paths['dlwmls'], 'DLWMLS_Volumes.csv')}"
                print(f"About to run: {post_dlwmls_cmd}")
                os.system(post_dlwmls_cmd)
                print("all done")

        out_csv = f"{st.session_state.paths['dlwmls']}/DLWMLS_Volumes.csv"
        num_dlwmls = utilio.get_file_count(st.session_state.paths["dlwmls"], ".nii.gz")
        if os.path.exists(out_csv):
            st.session_state.paths["csv_dlwmls"] = out_csv
            st.session_state.flags["csv_dlwmls"] = True
            st.success(
                f"DLWMLS images are ready ({st.session_state.paths['dlwmls']}, {num_dlwmls} scan(s))",
                icon=":material/thumb_up:",
            )


def panel_view() -> None:
    """
    Panel for viewing images
    """
    show_panel_view = st.checkbox(
        ":material/new_label: View Scans",
        disabled=not st.session_state.flags["csv_dlwmls"],
        value=False,
    )
    if not show_panel_view:
        return

    with st.container(border=True):
        # Selection of MRID
        try:
            df = pd.read_csv(st.session_state.paths["csv_dlwmls"])
            list_mrid = df.MRID.tolist()
        except:
            list_mrid = []
        if len(list_mrid) == 0:
            return

        sel_mrid = st.selectbox(
            "MRID", list_mrid, key="selbox_mrid", index=None, disabled=False
        )
        if sel_mrid is None:
            return

        # Create a list of checkbox options
        list_orient = st.multiselect(
            "Select viewing planes:", utilni.img_views, utilni.img_views, disabled=False
        )
        if list_orient is None or len(list_orient) == 0:
            return

        # View hide overlay
        is_show_overlay = st.checkbox("Show overlay", True, disabled=False)

        # Crop to mask area
        crop_to_mask = st.checkbox("Crop to mask", True, disabled=False)

        # Select images
        st.session_state.paths["sel_img"] = os.path.join(
            st.session_state.paths["FL"], sel_mrid + st.session_state.suff_flimg
        )
        if not os.path.exists(st.session_state.paths["sel_img"]):
            return

        st.session_state.paths["sel_seg"] = os.path.join(
            st.session_state.paths["dlwmls"], sel_mrid + st.session_state.suff_dlwmls
        )
        if not os.path.exists(st.session_state.paths["sel_seg"]):
            return

        list_rois = [1]

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
                            img, ind_view, mask_bounds[ind_view, :], tmp_orient, size_auto
                        )
                    else:
                        utilst.show_img3D(
                            img_masked, ind_view, mask_bounds[ind_view, :], tmp_orient, size_auto
                        )


def panel_download() -> None:
    """
    Panel for downloading results
    """
    if st.session_state.app_type == "cloud":
        show_panel_view = st.checkbox(
            ":material/new_label: Download Scans",
            disabled=not st.session_state.flags["csv_dlwmls"],
            value=False,
        )
        if not show_panel_view:
            return

        with st.container(border=True):

            # Zip results and download
            out_zip = bytes()
            if not False:
                if not os.path.exists(st.session_state.paths["download"]):
                    os.makedirs(st.session_state.paths["download"])
                f_tmp = os.path.join(st.session_state.paths["download"], "DLWMLS")
                out_zip = utilio.zip_folder(st.session_state.paths["dlwmls"], f_tmp)

            st.download_button(
                "Download DLWMLS results",
                out_zip,
                file_name=f"{st.session_state.dset}_DLWMLS.zip",
                disabled=False,
            )


st.divider()
panel_wdir()
panel_infl()
panel_dlwmls()
panel_view()
if st.session_state.app_type == "cloud":
    panel_download()

# FIXME: For DEBUG
utilst.add_debug_panel()
