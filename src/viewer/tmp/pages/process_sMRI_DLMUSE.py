import os
import re

import pandas as pd
import streamlit as st
import utils.utils_io as utilio
import utils.utils_nifti as utilni
import utils.utils_rois as utilroi
import utils.utils_st as utilst
from stqdm import stqdm


def panel_wdir():
    '''
    Panel for selecting the working dir
    '''
    icon = st.session_state.icon_thumb[st.session_state.flags['dir_out']]
    show_panel_wdir = st.checkbox(
        f":material/folder_shared: Working Directory {icon}",
        value = False
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

def panel_int1():
    '''
    Panel for uploading input t1 images
    '''
    st.session_state.flags['dir_t1'] = os.path.exists(st.session_state.paths['T1'])

    msg =  st.session_state.app_config[st.session_state.app_type]['msg_infile']
    icon = st.session_state.icon_thumb[st.session_state.flags['dir_t1']]
    show_panel_int1 = st.checkbox(
        f":material/upload: {msg} T1 Images {icon}",
        disabled = not st.session_state.flags['dir_out'],
        value = False
    )
    if not show_panel_int1:
        return

    with st.container(border=True):
        if st.session_state.app_type == "cloud":
            utilst.util_upload_folder(
                st.session_state.paths["T1"], "T1 images", False,
                "Nifti images can be uploaded as a folder, multiple files, or a single zip file"
            )
            fcount = utilio.get_file_count(st.session_state.paths["T1"])
            if fcount > 0:
                st.session_state.flags['T1'] = True
                st.success(
                    f"Data is ready ({st.session_state.paths["T1"]}, {fcount} files)",
                    icon=":material/thumb_up:"
                )

        else:  # st.session_state.app_type == 'desktop'
            utilst.util_select_folder(
                "selected_img_folder",
                "T1 nifti image folder",
                st.session_state.paths["T1"],
                st.session_state.paths["last_in_dir"],
                False,
            )
            fcount = utilio.get_file_count(st.session_state.paths["T1"])
            if fcount > 0:
                st.session_state.flags['dir_t1'] = True
                st.success(
                    f"Data is ready ({st.session_state.paths["T1"]}, {fcount} files)",
                    icon=":material/thumb_up:"
                )


def panel_dlmuse():
    '''
    Panel for running dlmuse
    '''
    icon = st.session_state.icon_thumb[st.session_state.flags['csv_dlmuse']]
    show_panel_dlmuse = st.checkbox(
        f":material/new_label: Run DLMUSE {icon}",
        disabled = not st.session_state.flags['dir_t1'],
        value = False
    )
    if not show_panel_dlmuse:
        return

    with st.container(border=True):
      # Device type
      if st.session_state.app_type == "CLOUD":
        device = 'cuda'
      else:
        helpmsg = "Choose 'cuda' if your computer has an NVIDIA GPU, 'mps' if you have an Apple M-series chip, and 'cpu' if you have a standard CPU."
        device = utilst.user_input_select(
            "Device",
            "key_select_device",
            ["cuda", "cpu", "mps"],
            None,
            helpmsg,
            False
        )

        # Button to run DLMUSE
        btn_seg = st.button("Run DLMUSE", disabled = False)
        if btn_seg:
            run_dir = os.path.join(st.session_state.paths["root"], "src", "NiChart_DLMUSE")
            if not os.path.exists(st.session_state.paths["dlmuse"]):
                os.makedirs(st.session_state.paths["dlmuse"])

            with st.spinner("Wait for it..."):
                dlmuse_cmd = f"NiChart_DLMUSE -i {st.session_state.paths['T1']} -o {st.session_state.paths['dlmuse']} -d {device} --cores 1"
                st.info(f"Running: {dlmuse_cmd}", icon=":material/manufacturing:")

                # FIXME : bypass dlmuse run
                print(f"About to run: {dlmuse_cmd}")
                os.system(dlmuse_cmd)

        out_csv = f"{st.session_state.paths['dlmuse']}/DLMUSE_Volumes.csv"
        num_dlmuse = utilio.get_file_count(st.session_state.paths["dlmuse"], '.nii.gz')
        if os.path.exists(out_csv):
            st.session_state.paths["csv_dlmuse"] = out_csv
            st.session_state.flags["csv_dlmuse"] = True
            st.success(
                f"DLMUSE images are ready ({st.session_state.paths['dlmuse']}, {num_dlmuse} scan(s))",
                icon=":material/thumb_up:",
        )

def panel_view():
    '''
    Panel for viewing images
    '''
    show_panel_view = st.checkbox(
        f":material/new_label: View Scans",
        disabled = not st.session_state.flags['csv_dlmuse'],
        value = False
    )
    if not show_panel_view:
        return

    with st.container(border=True):
        # Selection of MRID
        try:
            df = pd.read_csv(st.session_state.paths["csv_dlmuse"])
            list_mrid = df.MRID.tolist()
        except:
            list_mrid = []
        if len(list_mrid) == 0:
            st.warning('Result file is empty!')            
            return

        sel_mrid = st.selectbox(
            "MRID",
            list_mrid,
            key="selbox_mrid",
            index=None,
            disabled = False
        )
        if sel_mrid is None:
            st.warning('Please select the MRID!')            
            return

        # Create combo list for selecting target ROI
        list_roi_names = utilroi.get_roi_names(st.session_state.dicts["muse_sel"])
        sel_var = st.selectbox(
            "ROI",
            list_roi_names,
            key="selbox_rois",
            index=None,
            disabled = False
        )
        if sel_var is None:
            st.warning('Please select the ROI!')
            return

        # Create a list of checkbox options
        list_orient = st.multiselect(
            "Select viewing planes:",
            utilni.img_views,
            utilni.img_views,
            disabled = False
        )
        
        if list_orient is None or len(list_orient) == 0:
            st.warning('Please select the viewing plane!')            
            return

        # View hide overlay
        is_show_overlay = st.checkbox("Show overlay", True, disabled = False)

        # Crop to mask area
        crop_to_mask = st.checkbox("Crop to mask", True, disabled = False)

        # Get indices for the selected var
        list_rois = utilroi.get_list_rois(
            sel_var,
            st.session_state.rois['roi_dict_inv'],
            st.session_state.rois['roi_dict_derived'],
        )
        
        print('aAA')
        print(list_rois)
        
        if list_rois is None:
            st.warning('ROI list is empty!')            
            return

        # Select images
        if sel_mrid is None:
            st.warning('Please select the MRID!')            
            return

        st.session_state.paths["sel_img"] = os.path.join(
            st.session_state.paths["T1"], sel_mrid + st.session_state.suff_t1img
        )
        if not os.path.exists(st.session_state.paths["sel_img"]):
            st.warning(f'Could not locate underlay image: {st.session_state.paths["sel_img"]}')
            return

        st.session_state.paths["sel_seg"] = os.path.join(
            st.session_state.paths["dlmuse"], sel_mrid + st.session_state.suff_seg
        )
        if not os.path.exists(st.session_state.paths["sel_seg"]):
            st.warning(f'Could not locate overlay image: {st.session_state.paths["sel_seg"]}')
            return


        with st.spinner("Wait for it..."):
          # Select images
          flag_img = False
          ## FIXME: at least on cloud, can get here with multiple _T1 appended (see marked lines)
          flag_img = False
          if sel_mrid is not None:
            st.session_state.paths["sel_img"] = os.path.join(
                ## hardcoded fix for T1 suffix
                st.session_state.paths["T1"], re.sub(r"_T1$", "", sel_mrid)  + st.session_state.suff_t1img
            )
            st.session_state.paths["sel_seg"] = os.path.join(
                ## hardcoded fix for T1 suffix 
                st.session_state.paths["DLMUSE"], re.sub(r"_T1$", "", sel_mrid) + st.session_state.suff_seg
            )


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
                    if is_show_overlay is False:
                        utilst.show_img3D(
                            img, ind_view, mask_bounds[ind_view, :], tmp_orient
                        )
                    else:
                        utilst.show_img3D(
                            img_masked, ind_view, mask_bounds[ind_view, :], tmp_orient
                        )

def panel_download():
    '''
    Panel for downloading results
    '''
    if st.session_state.app_type == "cloud":
        show_panel_view = st.checkbox(
            f":material/new_label: Download Scans",
            disabled = not st.session_state.flags['csv_dlmuse'],
            value = False
        )
        if not show_panel_view:
            return

    with st.container(border=True):

        # Zip results and download
        out_zip = bytes()
        if not False:
            if not os.path.exists(st.session_state.paths["download"]):
                os.makedirs(st.session_state.paths["download"])
            f_tmp = os.path.join(st.session_state.paths["download"], "DLMUSE")
            out_zip = utilio.zip_folder(st.session_state.paths["dlmuse"], f_tmp)

        st.download_button(
            "Download DLMUSE results",
            out_zip,
            file_name=f"{st.session_state.dset}_DLMUSE.zip",
            disabled=False,
        )

st.markdown(
    """
    - Segmentation of T1-weighted MRI scans into anatomical regions of interest (ROIs)
    - [DLMUSE](https://github.com/CBICA/NiChart_DLMUSE): Fast deep learning based segmentation into 145 ROIs + 105 composite ROIs
        """
)

st.divider()
panel_wdir()
panel_int1()
panel_dlmuse()
panel_view()
if st.session_state.app_type == "cloud":
    panel_download()

# FIXME: For DEBUG
utilst.add_debug_panel()
