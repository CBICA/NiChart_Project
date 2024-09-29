import streamlit as st
import os
from math import ceil
import nibabel as nib
import numpy as np
from nibabel.orientations import axcodes2ornt, ornt_transform

VIEWS = ["axial", "sagittal", "coronal"]
VIEW_AXES = [0, 2, 1]
VIEW_OTHER_AXES = [(1,2), (0,1), (0,2)]
MASK_COLOR = (0, 255, 0)  # RGB format
MASK_COLOR = np.array([0.0, 1.0, 0.0])  # RGB format
OLAY_ALPHA = 0.2

def reorient_nifti(nii_in, ref_orient = 'LPS'):
    '''
    Initial img is reoriented to a standard orientation
    '''

    # Find transform from current (approximate) orientation to
    # target, in nibabel orientation matrix and affine forms
    orient_in = nib.io_orientation(nii_in.affine)
    orient_out = axcodes2ornt(ref_orient)
    transform = ornt_transform(orient_in, orient_out)

    # Apply transform
    nii_reorient = nii_in.as_reoriented(transform)

    # Return reoriented image
    return nii_reorient

def crop_image(img, mask):
    '''
    Crop img to the foreground of the mask
    '''

    # Detect bounding box
    nz = np.nonzero(mask)
    mn = np.min(nz, axis=1)
    mx = np.max(nz, axis=1)

    # Calculate crop to make all dimensions equal size
    mask_sz = mask.shape
    crop_sz = mx - mn 
    new_sz = max(crop_sz)
    pad_val = (new_sz - crop_sz) // 2
    
    min1 = mn - pad_val
    max1 = mx + pad_val

    min2 = np.max([min1, [0,0,0]], axis=0)
    max2 = np.min([max1, mask_sz], axis=0)
    
    pad1 = list(np.max([min2-min1, [0,0,0]], axis=0))
    pad2 = list(np.max([max1-max2, [0,0,0]], axis=0))
    
    # Crop image
    img = img[min2[0]:max2[0], min2[1]:max2[1], min2[2]:max2[2]]
    mask = mask[min2[0]:max2[0], min2[1]:max2[1], min2[2]:max2[2]]

    # Pad image
    padding = np.array([pad1, pad2]).T
    
    if padding.sum() > 0:
        img = np.pad(img, padding, mode='constant', constant_values=0)
        mask = np.pad(mask, padding, mode='constant', constant_values=0)
    
    return img, mask

def detect_mask_bounds(mask):
    '''
    Detect the mask start, end and center in each view
    Used later to set the slider in the image viewer
    '''

    mask_bounds = np.zeros([3,3]).astype(int)
    for i, axis in enumerate(VIEW_AXES):
        mask_bounds[i,0] = 0
        mask_bounds[i,1] = mask.shape[i]
        slices_nz = np.where(np.sum(mask, axis = VIEW_OTHER_AXES[i]) > 0)[0]
        try:
            mask_bounds[i,2] = slices_nz[len(slices_nz) // 2]
        except:
            # Could not detect masked region. Set center to image center
            mask_bounds[i,2] = mask.shape[i] // 2

    return mask_bounds

@st.cache_data
def prep_image_and_olay(f_img, f_mask, sel_var_ind, dict_derived):
    '''
    Read images from files and create 3D matrices for display
    '''

    # Read nifti
    nii_img = nib.load(f_img)
    nii_mask = nib.load(f_mask)

    # Reorient nifti
    nii_img = reorient_nifti(nii_img, ref_orient = 'IPL')
    nii_mask = reorient_nifti(nii_mask, ref_orient = 'IPL')

    # Extract image to matrix
    img = nii_img.get_fdata()
    mask = nii_mask.get_fdata()

    # Convert image to uint
    img = (img.astype(float) / img.max())

    # Crop image to ROIs and reshape
    img, mask = crop_image(img, mask)

    # Select target roi: derived roi
    list_rois = dict_derived[sel_var_ind]
    mask = np.isin(mask, list_rois)

    # # Select target roi: single roi
    # mask = (mask == sel_var_ind).astype(int)

    # Merge image and mask
    img = np.stack((img,)*3, axis=-1)

    img_masked = img.copy()
    img_masked[mask == 1] = (img_masked[mask == 1] * (1 - OLAY_ALPHA) + MASK_COLOR * OLAY_ALPHA)

    return img, mask, img_masked

