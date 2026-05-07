from typing import Any

import numpy as np
import nibabel as nib
from nibabel.orientations import axcodes2ornt, ornt_transform
from scipy import ndimage
import gui.utils_widgets as utilwd
import os

import streamlit as st
from stqdm import stqdm

from utils.utils_logger import setup_logger
logger = setup_logger()

## Constants
img_views = ["axial", "coronal", "sagittal"]
VIEW_AXES = [0, 1, 2]
VIEW_OTHER_AXES = [(1, 2), (0, 2), (0, 1)]
MASK_COLOR = np.array([0.0, 1.0, 0.0])  # RGB format
OLAY_ALPHA = 0.2

MAP_COLOR = np.array([1.0, 0.0, 0.0])  # RGB format
MAP_ALPHA = 0.7

def pad_image(img: np.ndarray) -> np.ndarray:
    """
    Pad img to equal x,y,z
    """

    # Detect max size
    simg = img.shape
    mx = np.max(simg)

    # Calculate padding values to make dims equal
    pad_vals = (mx - np.array(simg)) // 2

    # Create padded image
    out_img = np.zeros([mx, mx, mx])

    # Insert image inside the padded image
    out_img[
        pad_vals[0] : pad_vals[0] + simg[0],
        pad_vals[1] : pad_vals[1] + simg[1],
        pad_vals[2] : pad_vals[2] + simg[2],
    ] = img

    return out_img

def reorient_nifti(nii_in: Any, ref_orient: str = "LPS") -> Any:
    """
    Initial img is reoriented to a standard orientation
    """

    # Find transform from current (approximate) orientation to
    # target, in nibabel orientation matrix and affine forms
    orient_in = nib.io_orientation(nii_in.affine)
    orient_out = axcodes2ornt(ref_orient)
    transform = ornt_transform(orient_in, orient_out)

    # Apply transform
    nii_reorient = nii_in.as_reoriented(transform)

    # Return reoriented image
    return nii_reorient

def crop_image(img: np.ndarray, mask: np.ndarray, crop_to_mask: bool) -> Any:
    """
    Crop img to the foreground of the mask
    """

    # Detect bounding box
    if crop_to_mask:
        nz = np.nonzero(mask)
    else:
        nz = np.nonzero(img)

    mn = np.min(nz, axis=1)
    mx = np.max(nz, axis=1)

    # Calculate crop to make all dimensions equal size
    mask_sz = mask.shape
    crop_sz = mx - mn
    new_sz = max(crop_sz)
    pad_val = (new_sz - crop_sz) // 2

    min1 = mn - pad_val
    max1 = mx + pad_val

    min2 = np.max([min1, [0, 0, 0]], axis=0)
    max2 = np.min([max1, mask_sz], axis=0)

    pad1 = list(np.max([min2 - min1, [0, 0, 0]], axis=0))
    pad2 = list(np.max([max1 - max2, [0, 0, 0]], axis=0))

    # Crop image
    img = img[min2[0] : max2[0], min2[1] : max2[1], min2[2] : max2[2]]
    mask = mask[min2[0] : max2[0], min2[1] : max2[1], min2[2] : max2[2]]

    # Pad image
    padding = np.array([pad1, pad2]).T

    if padding.sum() > 0:
        img = np.pad(img, padding, mode="constant", constant_values=0)
        mask = np.pad(mask, padding, mode="constant", constant_values=0)

    return img, mask

def detect_mask_bounds(mask: Any) -> Any:
    """
    Detect the mask start, end and center in each view
    Used later to set the slider in the image viewer
    """
    mask_bounds = np.zeros([3, 3]).astype(int)
    for i, _ in enumerate(VIEW_AXES):
        mask_bounds[i, 0] = 0
        mask_bounds[i, 1] = mask.shape[i]
        slices_nz = np.where(np.sum(mask, axis=VIEW_OTHER_AXES[i]) > 0)[0]
        try:
            mask_bounds[i, 2] = slices_nz[len(slices_nz) // 2]
        except:
            # Could not detect masked region. Set center to image center
            mask_bounds[i, 2] = mask.shape[i] // 2

    return mask_bounds

def detect_img_bounds(img: np.ndarray) -> np.ndarray:
    """
    Detect the img start, end and center in each view
    Used later to set the slider in the image viewer
    """

    img_bounds = np.zeros([3, 3]).astype(int)
    for i, _ in enumerate(VIEW_AXES):
        img_bounds[i, 0] = 0
        img_bounds[i, 1] = img.shape[i]
        img_bounds[i, 2] = img.shape[i] // 2

    return img_bounds

def select_mriplot_settings(ptype):
    '''
    Panel to select mriplot settings
    '''
    ss = st.session_state
    pdict = ss[ptype]

    utilwd.my_multiselect(
        f"{ptype}.settings.mri_orient", img_views, 'View Planes'
    )

    mri_orient = pdict['settings']['mri_orient']
    if mri_orient is None or len(mri_orient) == 0:
        return

    utilwd.my_checkbox(
        f'{ptype}.settings.mri_flag_olay', "Show olay"
    )

    utilwd.my_checkbox(
        f'{ptype}.settings.mri_flag_crop', 'Crop to mask'
    )

@st.cache_data(max_entries=1)  # type:ignore
def prep_image_and_olay(
    f_img,
    f_mask,
    list_rois = None,
    minmax = [0,1],
    crop_to_mask = False,
) -> Any:
    """
    Read images from files and create 3D matrices for display
    """
    # Read nifti
    nii_img = nib.load(f_img)
    nii_mask = nib.load(f_mask)

    # Reorient nifti
    nii_img = reorient_nifti(nii_img, ref_orient="IPL")
    nii_mask = reorient_nifti(nii_mask, ref_orient="IPL")

    # Extract image to matrix
    out_img = nii_img.get_fdata()
    out_mask = nii_mask.get_fdata()

    # Rescale image and out_mask to equal voxel size in all 3 dimensions
    out_img = ndimage.zoom(out_img, nii_img.header.get_zooms(), order=0, mode="nearest")
    out_mask = ndimage.zoom(
        out_mask, nii_mask.header.get_zooms(), order=0, mode="nearest"
    )

    # Shift values in out_img to remove negative values
    out_img = out_img - np.min([0, out_img.min()])

    # Convert image to uint
    out_img = out_img.astype(float) / out_img.max()

    # Crop image to ROIs and reshape
    out_img, out_mask = crop_image(out_img, out_mask, crop_to_mask)

    min_val, max_val = minmax[0], minmax[1]

    # Create mask with sel roi
    if list_rois is not None:
        out_mask = np.isin(out_mask, list_rois)
    else:
        out_mask = ((out_mask >= min_val) & (out_mask <= max_val)).astype(int)
        st.write(f'Masked voxels: {(out_mask>0).sum()}')

    # Merge image and out_mask
    out_img = np.stack((out_img,) * 3, axis=-1)

    out_img_out_masked = out_img.copy()
    out_img_out_masked[out_mask == 1] = (
        # WARNING & FIXME:
        # @spirosmaggioros: I don't think this should be like this, something is wrong here with MASK_COLOR * OLAY_ALPHA
        out_img_out_masked[out_mask == 1] * (1 - OLAY_ALPHA)
        + MASK_COLOR * OLAY_ALPHA  # type:ignore
    )

    return out_img, out_mask, out_img_out_masked

@st.cache_data  # type:ignore
def prep_image(f_img: np.ndarray) -> np.ndarray:
    """
    Read image from file and create 3D matrice for display
    """

    # Read nifti
    nii_img = nib.load(f_img)

    # Reorient nifti
    nii_img = reorient_nifti(nii_img, ref_orient="IPL")

    # Extract image to matrix
    out_img = nii_img.get_fdata()

    # Rescale image to equal voxel size in all 3 dimensions
    out_img = ndimage.zoom(out_img, nii_img.header.get_zooms(), order=0, mode="nearest")

    out_img = out_img.astype(float) / out_img.max()

    # Pad image to equal size in x,y,z
    out_img = pad_image(out_img)

    # Convert image to rgb
    out_img = np.stack((out_img,) * 3, axis=-1)

    return out_img

def show_img_slices(img, scroll_axis, sel_axis_bounds, orientation, wimg = None):
    """
    Display 3D mri img slice
    """
    # Create a slider to select the slice index
    slice_index = st.slider(
        f"{orientation}", 
        0,
        sel_axis_bounds[1] - 1,
        value=sel_axis_bounds[2],
        key=f"slider_{orientation}",
        # width = wimg
    )

    # Extract the slice and display it
    if wimg is None:
        if scroll_axis == 0:
            st.image(img[slice_index, :, :], width='stretch')
        elif scroll_axis == 1:
            st.image(img[:, slice_index, :], width='stretch')
        else:
            st.image(img[:, :, slice_index], width='stretch')
    else:
        if scroll_axis == 0:
            st.image(img[slice_index, :, :], width=wimg)
        elif scroll_axis == 1:
            st.image(img[:, slice_index, :], width=wimg)
        else:
            st.image(img[:, :, slice_index], width=wimg)

def panel_view_seg(ptype):
    '''
    Panel to display segmented image overlaid on underlay image
    '''    
    logger.debug('Panel: View Segmentation')

    ss = st.session_state
    pdict = ss[ptype]

    mrid = pdict['selected']['mrid']
    ylab = pdict['selected']['ylab']

    if mrid is None:
        return

    if ylab is None:
        return

    # Find roi indices
    col_dict = pdict['data']['col_dict']
    roi = col_dict.renamed_to_roi_index(ylab)
    if roi is None:
        return
    roi_indices = st.session_state.dicts['muse']['derived'].get(roi, [roi])
    if roi_indices is None or len(roi_indices)<=0:
        return
    
    ## FIXME
    in_dir = st.session_state.paths['prj_dir']
    ulay = os.path.join(
        in_dir, 't1', f'{mrid}_T1.nii.gz'
    )
    olay = os.path.join(
        in_dir, 'dlmuse-seg', f'{mrid}_T1_DLMUSE.nii.gz'
    )
    if not os.path.exists(ulay):
        return
    if not os.path.exists(olay):
        return
    pdict['selected']['mri_ulay'] = ulay
    pdict['selected']['mri_olay'] = olay
    ##

    tab1, tab2 = st.tabs(
        [':material/visibility_off:', 'Options'],
        on_change='rerun',
        key='_tabs_mri_controls',
    )

    if tab2.open:
        with tab2:
            with st.container(border=True):
                select_mriplot_settings(ptype)

    if pdict['selected']['mri_ulay'] is None:
        st.toast('Underlay image not found!')
        return

    if pdict['selected']['mri_olay'] is None:
        st.toast('Overlay image not found!')
        pdict['settings']['mri_flag_olay'] = False
        st.rerun()

    # Show images
    with st.container(border=True):
        st.markdown(f'Subject: `{mrid}`')
        try:
            img, mask, img_masked = prep_image_and_olay(
                pdict['selected']['mri_ulay'],
                pdict['selected']['mri_olay'],
                list_rois=roi_indices,
                crop_to_mask=pdict['settings']['mri_flag_crop']
            )
        except Exception as exc:
            st.warning(f'Could not read image files! {exc}')
            st.write(pdict['selected']['mri_ulay'])
            st.write(pdict['selected']['mri_olay'])
            st.write(pdict['settings'])
            return
        
        img_bounds = detect_mask_bounds(mask)

        nviews = len(pdict['settings']['mri_orient'])
        if nviews == 0:
            return
        cols = st.columns(nviews)
        for i, sel_orient in enumerate(pdict['settings']['mri_orient']):
            with cols[i]:
                ind_view = img_views.index(sel_orient)
                if pdict['selected']['mri_olay'] is None or pdict['settings']['mri_flag_olay'] is False:
                    show_img_slices(
                        img, ind_view, img_bounds[ind_view, :], sel_orient
                    )
                else:
                    show_img_slices(
                        img_masked, ind_view, img_bounds[ind_view, :], sel_orient
                    )

def panel_view_map():
    '''
    Panel to display a map with float values overlaid on underlay image
    '''
    ss = st.session_state
    
    if ss.plots_mri_ulay is None:
        return

    if ss.plots_mri_olay is None:
        return

    # Show images
    with st.container(border=True):
        with st.spinner("Wait for it..."):
            # Process image (and mask) to prepare final 3d matrix to display
            
            #try:
            img, mask, img_masked = prep_image_and_olay(
                ss.plots_mri_ulay, 
                ss.plots_mri_olay, 
                minmax = ss.plots_mri_map_minmax, 
                crop_to_mask = ss.plots_mri_flag_crop
            )
            #except Exception as e:
                #st.warning(f'Could not read image files!')
                #st.write(params['ulay'])
                #st.write(params['olay'])
                #return
            
            img_bounds = detect_mask_bounds(mask)

            cols = st.columns(3)
            for i, tmp_orient in stqdm(
                enumerate(ss.plots_mri_orient),
                desc="Showing images ...",
                total=len(ss.plots_mri_orient)
            ):
                with cols[i]:
                    ind_view = img_views.index(tmp_orient)
                    if ss.plots_mri_olay is None or ss.plots_mri_flag_olay is False:
                        show_img_slices(
                            img, ind_view, img_bounds[ind_view, :], tmp_orient
                        )
                    else:
                        show_img_slices(
                            img_masked, ind_view, img_bounds[ind_view, :], tmp_orient
                        )

def panel_view_mri_simple(fname):
    '''
    Panel for viewing a nifti scan
    '''
    try:
        # Prepare and show axial slice of the image
        img = prep_image(fname)
        img_bounds = detect_img_bounds(img)
        ind_view = img_views.index('axial')
        size_auto = True
        show_img_slices(img, ind_view, img_bounds[ind_view, :], 'axial', 500)
    except:
        st.warning(":material/thumb_down: Could not open image")

