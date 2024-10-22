from typing import Any

import nibabel as nib
import numpy as np
import streamlit as st
from nibabel.orientations import axcodes2ornt, ornt_transform
from scipy import ndimage

VIEWS = ["axial", "coronal", "sagittal"]
VIEW_AXES = [0, 1, 2]
VIEW_OTHER_AXES = [(1, 2), (0, 2), (0, 1)]
MASK_COLOR = (0, 255, 0)  # RGB format
MASK_COLOR = np.array([0.0, 1.0, 0.0])  # RGB format
OLAY_ALPHA = 0.2


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


def detect_mask_bounds(mask: Any) -> Any:
    """
    Detect the mask start, end and center in each view
    Used later to set the slider in the image viewer
    """

    mask_bounds = np.zeros([3, 3]).astype(int)
    for i, axis in enumerate(VIEW_AXES):
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
    for i, axis in enumerate(VIEW_AXES):
        img_bounds[i, 0] = 0
        img_bounds[i, 1] = img.shape[i]
        img_bounds[i, 2] = img.shape[i] // 2

    return img_bounds


@st.cache_data  # type:ignore
def prep_image_and_olay(
    f_img: np.ndarray, f_mask: Any, list_rois: list, crop_to_mask: bool
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

    # Create mask with sel roi
    out_mask = np.isin(out_mask, list_rois)

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


def check_roi_index(f_img: str, roi_ind: int) -> bool:
    """
    Check if index is one of the labels in the seg img
    """
    try:
        # Read nifti
        nii_img = nib.load(f_img)

        # Extract image to matrix
        out_img = nii_img.get_fdata().flatten()

        # Get labels
        list_lbl = np.unique(out_img)

        return roi_ind in list_lbl

    except:
        return False
