import numpy as np
import nibabel as nib
from scipy import ndimage
from scipy.ndimage.measurements import label

from NiChart_DLMUSE.CombineMasks import calc_bbox_with_padding


###---------mask image-----------
def apply_mask(in_img_name, mask_img_name, out_img_name):
    ## Read input image and mask
    nii_in = nib.load(in_img_name)
    nii_mask = nib.load(mask_img_name)

    img_in = nii_in.get_fdata()
    img_mask = nii_mask.get_fdata()

    ## Mask image
    img_in[img_mask == 0] = 0

    ################################
    ## INFO: nnunet hallucinated on images with large FOV. To solve this problem
    ##       we added pre/post processing steps to crop initial image around ICV 
    ##       mask before sending to DLMUSE
    ##
    ## Crop image
    bcoors = calc_bbox_with_padding(img_mask)
    img_in_crop = img_in[bcoors[0,0]:bcoors[0,1], bcoors[1,0]:bcoors[1,1], bcoors[2,0]:bcoors[2,1]]    

    ## Save out image
    nii_out = nib.Nifti1Image(img_in_crop, nii_in.affine, nii_in.header)    
    nii_out.to_filename(out_img_name)